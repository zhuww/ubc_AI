#!/usr/bin/env python

"""
Aaron Berndsen: A GUI to view and rank PRESTO candidates.

Requires a GUI input file with (at least) one column of the
candidate filename/location (can be .pfd, .ps, or .png).
Subsequent columns are the AI and user votes.

pfd: use of these files requires PRESTO's show_pfd command
     We search for it, but you may hard-code the location below
     Allows for AI_view, which downsamples and pca's the data
     to the same form that typical AI algorithms use.
ps: requires either PythonMagick, or system calls to ImageMagick
png: quick to display

"""
import atexit
import numpy as np
import os
import shutil
import subprocess
import sys
import tempfile

from gi.repository import Gtk, Gdk
import pylab as plt

#next taken from ubc_AI.training and ubc_AI.samples
from training import pfddata
from sklearn.decomposition import RandomizedPCA as PCA

try:
    from PythonMagick import Image
    pyimage = True
except ImportError:
    print "\t Install PythonMagick or imagemagick if inputing postscript files"
    pyimage = False

#PRESTO's show_pfd command:
show_pfd = False
for p in os.environ.get('PATH').split(':'):
    cmd = '%s/show_pfd' % p
    if os.path.exists(cmd):
        show_pfd = cmd
        break
if not show_pfd:
    print "\tCouldn't find PRESTO's show_pfd executable"
    print "\t This will limit functionality"

#iter on each "n". auto-save after every 10
cand_view = 0
#store AI_view png's in a temporary dir
tempdir = tempfile.mkdtemp(prefix='AIview_')
atexit.register(lambda: shutil.rmtree(tempdir))
bdir = '/'.join(__file__.split('/')[:-1])

class MainFrameGTK(Gtk.Window):
    """This is the Main Frame for the GTK application"""
    
    def __init__(self, data=None):
        Gtk.Window.__init__(self, title='pfd viewer')

        self.gladefile = "%s/pfdviewer.glade" % bdir
        self.builder = Gtk.Builder()
        self.builder.add_from_file(self.gladefile)

        ## glade-related objects
        self.voterbox = self.builder.get_object("voterbox")
        self.pfdwin = self.builder.get_object("pfdwin")
        self.builder.connect_signals(self)
        self.pfdwin.show_all()
        self.listwin = self.builder.get_object('listwin')
        self.listwin.show_all()
        self.statusbar = self.builder.get_object('statusbar')
        self.pfdtree = self.builder.get_object('pfdtree')
        self.pfdstore = self.builder.get_object('pfdstore')
        self.image = self.builder.get_object('image')
        self.image_disp = self.builder.get_object('image_disp')
        self.autosave = self.builder.get_object('autosave')
        self.pfdtree.connect("cursor-changed",self.on_pfdtree_select_row)
        self.aiview = self.builder.get_object('aiview')
        self.aiview_win = self.builder.get_object('aiview_win')
        self.aiview_win.set_deletable(False)
        self.aiview_win.connect('delete-event', lambda w, e: w.hide() or True)


#allow Ctrl+s like key-functions        
        self.modifier = None
#where are pfd/png/ps files stored
        self.basedir = '.'
#AI prob is always 2nd col in GUI, use it to sort
        for vi, v  in enumerate(['fname','AI prob', 'voter prob']):
            #only show two columns at a time
            cell = Gtk.CellRendererText()
            col = Gtk.TreeViewColumn(v, cell, text=vi)
            col.set_property("alignment", 0.5)
            col.set_max_width(150)
            self.pfdtree.append_column(col)
        self.pfdstore.set_sort_column_id(1,1)


        ## data-analysis related objects
        self.voters = []
        self.savefile = None
        self.loadfile = None 
        #if we were passed a data file, read it in
        if data != None:
            self.on_open(event='load', fin=data)
        else:
            self.data = None
        # start with default and '<new>' voters
        if self.data != None:
            self.voters = list(self.data.dtype.names[1:])
            if 'AI' in self.voters:
                AIi = self.voters.index('AI')
                self.voters.pop(AIi)
            if '<new>' not in self.voters:
                self.voters.insert(0,'<new>')
            self.active_voter = 1
        
        ##GUI init options
            #done in on_open
# set default voter to 1st non-AI
#            print "AAAR",self.voters
#            for v in self.voters:
#                if v != 'AI':
#                    self.voterbox.append_text(v)
            if len(self.voters) > 2:
                self.active_voter = 1
                self.voterbox.set_active(self.active_voter)
            self.dataload_update()
        #put cursor on first col. if there is data
            self.pfdtree.set_cursor(0)
            self.update()

        else:
            self.update('Please load a data file')
            self.voters = []
            self.active_voter = None

#keep track of the AI view files created (so we don't need to generate them)
        self.AIviewfiles = {}

############################
## data-manipulation actions

    def on_pfdwin_key_press_event(self, widget, event):
        """
        controls keypresses on over-all window

        """
        global cand_view
        key = Gdk.keyval_name(event.keyval)
        ctrl = event.state &\
            Gdk.ModifierType.CONTROL_MASK
        if self.active_voter:
            act_name = self.voters[self.active_voter]
        else:
            act_name = 'AI'

        #keep modifier keys until they are released
        if key in ['Control_L','Control_R','Alt_L','Alt_R']:
            self.modifier = key

        if key == 'q' and self.modifier in\
                ['Control', 'Control_L', 'Control_R']:
            self.on_menubar_delete_event(widget, event)

        elif key == 's' and self.modifier in ['Control_L', 'Control_R']:
            self.on_save(widget)
            pass
            
        elif key == 'l' and self.modifier in ['Control_L', 'Control_R']:
            self.on_open()

        elif key == 'n':           
            self.pfdtree_next()
        elif key == 'b':
            self.pfdtree_prev()

        #data-related (needs to be loaded)
        if self.data != None:
            if key == '0':
                if act_name != 'AI':
                    self.pfdstore_set_value(float(key))
                else:
                    self.statusbar.push(0,'Note: AI not editable')
                self.pfdtree_next()
            elif key == '1':
                if act_name != 'AI':
                    self.pfdstore_set_value(float(key))
                else:
                    self.statusbar.push(0,'Note: AI not editable')
                self.pfdtree_next()

            if key == '0' or key == '1':
                cand_view += 1
                if cand_view//10 == 1:
                    if self.autosave.get_active():
                        self.on_save()
                    else:
                        self.statusbar.push(0,'Remember to save your output')
            

    def dataload_update(self):
        """
        update the pfdstore whenever we load in a new data file
        
        set columns to "fname, AI, [1st non-AI voter... if exists]"

        """
        if self.active_voter:
            act_name = self.voters[self.active_voter]
#update the Treeview... 
            if self.data != None:
#turn off the model first for speed-up
                self.pfdtree.set_model(None)
                self.pfdstore.clear()
                for v in self.data[['fname','AI',act_name]]:
                    self.pfdstore.append(v)

                self.pfdtree.set_model(self.pfdstore)


    def on_pfdtree_select_row(self, widget, event=None):#, data=None):
        """
        responds to keypresses/cursor-changes in the pfdtree view,
        placing the appropriate candidate plot in the pfdwin.

        We also look for the .pfd or .ps files and convert them
        to png if necessary... though we make system calls for this

        if ai_view is active, we make a plot of data downsamples 
        to the AI and show that. If any of the AIview params are illegal
        we take (nbins, n_pca_comp) = (32, 0)... no pca

        """
        gsel = self.pfdtree.get_selection()
        if gsel:
            tmpstore, tmpiter = gsel.get_selected()
        else:
            tmpiter = None

        ncol = self.pfdstore.get_n_columns()
        if tmpiter != None:
            
            fname = '%s/%s' % (self.basedir,tmpstore.get_value(tmpiter, 0))

            #we are not doing "AI view" of data
            if not self.aiview.get_active():
                if fname.endswith('.ps'):
                    fpng = fname.replace('.ps','.png')
    # double-check ps file exists,
    # set fname to pfd file if not to regenerate it
                    if not os.path.exists(fname):
                        #strip .ps
                        pfdfile = os.path.splitext(fname)[0]
                        if os.path.exists(pfdfile):
                            fname = pfdfile
                            fpng = convert(fname)
                elif fname.endswith('.pfd'):
                    fpng = fname + '.png'
                elif fname.endswith('.png') or fname.endswith('.jpg'):
                    fpng = fname
                else:
                    note = "Don't recognize filetype %s" % fname
                    print note
                    self.statusbar.push(0, note)
                    fpng = ''

    #see if png exists locally already, otherwise generate it
                if not os.path.exists(fpng):
                    #convert to png (convert accepts .ps, or .pfd file)
                    fpng = convert(fname)
                if fpng and os.path.exists(fpng):
                    self.image.set_from_file(fpng)
                    self.image_disp.set_text('displaying : %s' % fname)
                else:
                    note = "Failed to generate png file %s" % fname
                    print note
                    self.statusbar.push(0,note)
                    self.image.set_from_file('')

                l = fname
                for i in range(1,ncol):
                    l += ' %s ' % tmpstore.get_value(tmpiter, i)
                print "active row",l

            else:

                #we are doing the AI view of the data
                fpng= ''
                if os.path.exists(fname) and fname.endswith('.pfd'):
                    self.statusbar.push(0,'Generating AIview...')

                    #have we generated this AIview before?
                    fpng = self.check_AIviewfile_match(fname)
                    if not fpng:
                        fpng = self.generate_AIviewfile(fname)

                if fpng and os.path.exists(fpng):
                    self.image.set_from_file(fpng)
                    self.image_disp.set_text('displaying : %s' % fname)
                else:
                    note = "Failed to generate png file %s" % fname
                    print note
                    self.statusbar.push(0,note)
                    self.image.set_from_file('')
                    self.image_disp.set_text('displaying : %s' % fname)

    def generate_AIviewfile(self, fname):
        """
        Given some PFD file and the AI_view flag, generate the png
        file for this particular view. 

        We save the files in /tmp/AIview*, 

        """
        
        pfd = pfddata(fname)
        plt.figure(figsize=(8,8))
        vals = [('pprof_nbins', 'pprof_pcacomp'), #pulse profile
                ('si_nbins', 'si_pcacomp'),       #frequency subintervals
                ('pi_bins', 'pi_pcacomp'),        #pulse intervals
                ('dm_bins', 'dm_pcacomp')         #DM-vs-chi2
                ]
        AIview = []
        for subplt, inp in enumerate(vals):
            nbins = self.builder.get_object(inp[0]).get_text()
            npca_comp = self.builder.get_object(inp[1]).get_text()
            try:
                nbins = int(nbins)
            except ValueError:
                nbins = 32
            try:
                npca_comp = int(npca_comp)
            except ValueError:
                npca_comp = 0 #no pca
            AIview.append([nbins, npca_comp])
            ax = plt.subplot(2, 2, subplt+1)
            if subplt == 0:
                data = pfd.getdata(phasebins=nbins)
                if npca_comp:
                    pca = PCA(n_components=npca_comp).fit(data)
                    pcadata = pca.transform(data)
                    data = pca.inverse_transform(pcadata)
                ax.plot(data)
                ax.set_title('pulse profile (bins, pca) = (%s,%s)'%(nbins,npca_comp))
            elif subplt == 1:
                data = pfd.getdata(subbands=nbins)
                if npca_comp:
                    pca = PCA(n_components=npca_comp).fit(data)
                    pcadata = pca.transform(data)
                    data = pca.inverse_transform(pcadata)
                ax.imshow(data.reshape(nbins, nbins),
                          cmap=plt.cm.gray)
                ax.set_title('subbands (bins, pca) = (%s,%s)'%(nbins,npca_comp))
            elif subplt == 2:
                data = pfd.getdata(intervals=nbins)
                if npca_comp:
                    pca = PCA(n_components=npca_comp).fit(data)
                    pcadata = pca.transform(data)
                    data = pca.inverse_transform(pcadata)
                ax.imshow(data.reshape(nbins,nbins),
                                cmap=plt.cm.gray)
                ax.set_title('intervals (bins, pca) = (%s,%s)'%(nbins,npca_comp))
            elif subplt == 3:
                data = pfd.getdata(DMbins=nbins)
                if npca_comp:
                    pca = PCA(n_components=npca_comp).fit(data)
                    pcadata = pca.transform(data)
                    data = pca.inverse_transform(pcadata)
                ax.plot(data)
                ax.set_title('DM curve (bins, pca) = (%s,%s)'%(nbins,npca_comp))
            ax.set_yticklabels([])
            ax.set_xticklabels([])

        fd, fpng = tempfile.mkstemp(dir=tempdir, suffix='.png')
        plt.savefig(fpng)
        #keep track of which AI views have been generated already
        if not self.AIviewfiles.has_key(fname):
            #I like my dictionaries: store the AIview for 
            #each pfd in a dictionary of dictionaries
            self.AIviewfiles[fname] = {}
            self.AIviewfiles[fname][str(AIview)] = fpng
        else:
            self.AIviewfiles[fname][str(AIview)] =  fpng
        return fpng

    def check_AIviewfile_match(self, fname):
        """
        read the AIview window and compare to our generated files.
        
        return: None if we haven't generated this view
                filename if we have
        
        """           
        vals = [('pprof_nbins', 'pprof_pcacomp'), #pulse profile
                ('si_nbins', 'si_pcacomp'),       #frequency subintervals
                ('pi_bins', 'pi_pcacomp'),        #pulse intervals
                ('dm_bins', 'dm_pcacomp')         #DM-vs-chi2
                ]
                    #have we generated this AIview before?
        AIview = []
        for  inp in vals:
            nbins = self.builder.get_object(inp[0]).get_text()
            npca_comp = self.builder.get_object(inp[1]).get_text()
            try:
                nbins = int(nbins)
            except ValueError:
                nbins = 32
            try:
                npca_comp = int(npca_comp)
            except ValueError:
                npca_comp = 0 #no pca
            AIview.append([nbins, npca_comp])

        fpng = ''
        if self.AIviewfiles.has_key(fname):
            if self.AIviewfiles[fname].has_key(str(AIview)):
                fpng = self.AIviewfiles[fname][str(AIview)]
        return fpng

        
    def pfdstore_set_value(self, value):
        """
        update the pfdstore value for the active user

        """
        (model, pathlist) = self.pfdtree.get_selection().get_selected_rows()
#only use first selected object
        if len(pathlist) == 0:
            #nothing selected, so go back to first
            self.pfdtree.set_cursor(0)
            (model, pathlist) = self.pfdtree.get_selection().get_selected_rows()
        path = pathlist[0]
        tree_iter = model.get_iter(path)

#update self.data (since dealing with TreeStore blows my mind)
        fname, x, oval = self.pfdstore[tree_iter]
        self.pfdstore[tree_iter][2] = value
        idx = self.data['fname'] == fname
        if self.active_voter:
            act_name = act_name = self.voters[self.active_voter]
            self.data[act_name][idx] = value
       

    def on_aiview_toggled(self, event):
        """
        display or destroy the AI_view parameters window

        """
        if self.aiview.get_active():
            self.aiview_win.show_all()
        else:
            self.aiview_win.hide()
# redraw the pfdwin 
        self.on_pfdtree_select_row(event)

        
    def on_aiview_change(self, event):
        """
        respond to any changes on the aiview plot params.
        set to red if any is wrong.

        """
        vals = ['pprof_nbins', 'pprof_pcacomp', #pulse profile
                'si_nbins', 'si_pcacomp',       #frequency subintervals
                'pi_bins', 'pi_pcacomp',        #pulse intervals
                'dm_bins', 'dm_pcacomp'         #DM-vs-chi2
                ]
        red = Gdk.color_parse("#FF0000")
        black = Gdk.color_parse("#000000")
        for v in vals:
            o = self.builder.get_object(v)
            value = o.get_text()
            try:
                t = int(o.get_text())
                o.modify_fg(Gtk.StateType.NORMAL, black)
            except ValueError:
                o.modify_fg(Gtk.StateType.NORMAL, red)
        #redraw the pfdwin
        self.on_pfdtree_select_row(event)

    def on_votevalue_changed(self, widget, event):
        """
        spinner has changed. get value and active object and update the
        value

        """
        self.onpfdwin_key_press_event(self, widget, event)


    def on_voterbox_changed(self, event=None):
        """
        read the voter box and change the active voter

        """
        curpos = self.pfdtree.get_cursor()[0]

        prev_voter = self.active_voter
        voter = self.voterbox.get_active_text()
        if voter == '<new>':
            #create a dialog for a new voter
            d = ''
            while d == '':
                d = inputbox('pfdviewer','choose your voting name')
            if d != None:
                if d not in self.voters:
                    print "adding voter data for %s" % d
                    self.voters.append(d)
                    self.data = add_voter(d, self.data)
                    self.voterbox.append_text(d)
                    self.active_voter = len(self.voters) - 1
                else:
                    note = 'User already exists. switching to it'
                    print note
                    self.statusbar.push(0,note)
                    self.active_voter = self.voters.index(d)
                    self.voterbox.set_active(self.active_voter)
            else:
                #return to previous state
                self.voterbox.set_active(self.active_voter)
        else:
            self.active_voter = self.voterbox.get_active() #get newest selection

        if prev_voter != self.active_voter:
            self.voterbox.set_active(self.active_voter)
            self.dataload_update()


############################
## menu-related actions

    def on_menubar_delete_event(self, widget, event=None):

        dlg = Gtk.MessageDialog(self, 0, Gtk.MessageType.INFO,
                                Gtk.ButtonsType.OK_CANCEL,
                                "Are you sure you want to quit the PFDviewer?")
        response = dlg.run()
        
        if response == Gtk.ResponseType.OK:
            print "Goodbye"
            dlg.destroy()
            Gtk.main_quit()
        dlg.destroy()

    def on_delete_win(self, widget, event=None):
        """
        respond to window close

        """
        print "Exiting... Goodbye"
        Gtk.main_quit()


    def on_open(self, event=None, fin=None):
        """
        load a data file. 
        Should contain two columns: filename AI_prob

        """
        if event != 'load':                           
            dialog = Gtk.FileChooserDialog("choose a file to load", self,
                                           Gtk.FileChooserAction.OPEN,
                                           (Gtk.STOCK_CANCEL, 
                                            Gtk.ResponseType.CANCEL,
                                            Gtk.STOCK_OPEN, Gtk.ResponseType.OK))
            response = dialog.run()
            if response == Gtk.ResponseType.OK:
                print "Select clicked"
                fname = dialog.get_filename()
                print "File selected: " + fname
            elif response == Gtk.ResponseType.CANCEL:
                print "Cancel clicked"
                fname = None
            dialog.destroy()
        else:
            fname = fin
                
        if fname:
            self.loadfile = fname
            self.data = load_data(fname)
            oldvoters = self.voters
            self.voters = list(self.data.dtype.names[1:]) #0=fnames
            AIi = self.voters.index('AI')
            self.voters.pop(AIi)
            if '<new>' not in self.voters:
                self.voters.insert(0,'<new>')
            self.active_voter = 1

            #add new voters to the voterbox
            for v in self.voters:
                if v not in oldvoters:
                    self.voterbox.append_text(v)

            if self.active_voter == None: 
                self.active_voter = 1
            elif self.active_voter >= len(self.voters):
                self.active_voter = len(self.voters)

            self.voterbox.set_active(self.active_voter)
        self.dataload_update()
        self.pfdtree.set_cursor(0)


    def on_save(self, event=None):
        """
        save the data

        """
        if self.savefile == None:
            dialog = Gtk.FileChooserDialog("choose an output file.", self,
                                           Gtk.FileChooserAction.SAVE,
                                           (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                                            Gtk.STOCK_OPEN, Gtk.ResponseType.OK))
            if self.loadfile:
                suggest = os.path.splitext(self.loadfile)[0] + '.npy'
            else:
                suggest = 'voter_rankings.npy'
            dialog.set_current_name(suggest)
            filter = Gtk.FileFilter()
            filter.set_name("numpy file (*.npy)")
            filter.add_pattern("*.npy")
            dialog.add_filter(filter)
            filter = Gtk.FileFilter()
            filter.set_name("text file (.dat)")
            filter.add_pattern("*.dat")
            dialog.add_filter(filter)
            filter = Gtk.FileFilter()
            filter.set_name("All *")
            filter.add_pattern("*")
            dialog.add_filter(filter)
            response = dialog.run()
            if response == Gtk.ResponseType.OK:
                self.savefile = dialog.get_filename()
                print "File selected: " + self.savefile
            dialog.destroy()

#        print "Writing data to %s" % self.savefile
        if self.savefile == None:
            note = "No file selected. File not saved"
            print note
            self.statusbar.push(0,note)
        elif self.savefile.endswith('.npy'):
            note = 'Saved to numpy file %s' % self.savefile
            np.save(self.savefile,self.data)
            if not self.autosave.get_active():
                print note
            self.statusbar.push(0,note)
        else:
            print "please consider saving to a .npy file"
            note = 'Saved to textfile %s' % self.savefile
            fout = open(self.savefile,'w')
            l1 = '#'
            l1 += ' '.join(self.data.dtype.names)
            l1 += '\n'
            fout.writelines(l1)
            for row in self.data:
                for r in row:
                    fout.write("%s " % r)
                fout.write("\n")
            fout.close()
            if not self.autosave.get_active():
                print note
            self.statusbar.push(0,note)

    def on_saveas(self, event=None):
        self.savefile = None
        self.on_save(event)

############################
## gui related actions

    def on_autosave_toggled(self, value):
        """

        """
        print "AUTO",self.autosave.get_active()


    def update(self, event=None):
        """
        update:
        *statusbar and make sure right cell is highlighted
        *voterbox to make sure it has all the right entries

        """        

        if self.data == None:
            ncand = 0
        else:
            ncand = len(self.data)

        if ncand == 0:
            stat = 'Please load a data file'
        else:
            stat = 'Loaded %s candidates' % ncand

#set cursor to first entry when loading a data file
        if event == 'load':
            self.pfdtree.set_cursor(0)
        self.statusbar.push(0, stat)


    def pfdtree_next(self):
        """
        select next row in pfdtree
        """
        (model, pathlist) = self.pfdtree.get_selection().get_selected_rows()
#only use first selected object
        path = pathlist[0]
        tree_iter = model.get_iter(path)
        npath = model.iter_next(tree_iter)
        if npath:
            nextpath = model.get_path(npath)
            self.pfdtree.set_cursor(nextpath) 

        
    def pfdtree_prev(self):
        """
        select prev row in pfdtree
        """
        (model, pathlist) = self.pfdtree.get_selection().get_selected_rows()
#only use first selected object
        path = pathlist[0]
        tree_iter = model.get_iter(path)
        prevn = model.iter_previous(tree_iter)
        if prevn:
            prevpath = model.get_path(prevn)
            self.pfdtree.set_cursor(prevpath)


    def on_pfdwin_key_release_event(self, widget, event):
        key = Gdk.keyval_name(event.keyval)
        if key not in ['Control_L','Control_R','Alt_L','Alt_R']:
            self.modifier = None

####### end MainFrame class ####

################################
## utilities

def convert(fin):
    """
    given a pfd or ps file, make the png file

    return:
    the name of the png file

    """
    global show_pfd
    fout = None
    if not os.path.exists(fin):
        print "Can't find file %s" % fin
        return
    if fin.endswith('.pfd'):
        #find PRESTO's show_pfd executable
        if not show_pfd:
            dialog = Gtk.FileChooserDialog("Locate show_pfd executable", self,
                                           Gtk.FileChooserAction.OPEN,
                                           (Gtk.STOCK_CANCEL, 
                                            Gtk.ResponseType.CANCEL,
                                            Gtk.STOCK_OPEN, Gtk.ResponseType.OK))
            response = dialog.run()
            if response == Gtk.ResponseType.OK:
                print "Select clicked"
                show_pfd = dialog.get_filename()
                print "File selected: " + fname
            elif response == Gtk.ResponseType.CANCEL:
                print "Cancel clicked"
        if show_pfd:
            #make the .ps file (converted, next, to .png)
            full_path = os.path.abspath(fin)
            basedir = '/'.join(full_path.split('/')[:-1])
            basename = os.path.basename(fin)
            cmd = [show_pfd, '-noxwin', full_path]
            subprocess.call(cmd, shell=False,
                            stdout=open('/dev/null','w'))
#delete the other generated files if they already exist
#otherwise move them to same location as pfd files
            fold = '%s/%s.ps' % (basedir, basename)
            if os.path.exists(fold):
               os.remove(fold)
            else:
                shutil.move('%s.ps' % basename, basedir)
            fb = '%s/%s.bestprof' % (basedir, basename)
            if os.path.exists(fb):
                os.remove(fold)
            else:
                shutil.move('%s.bestprof' % basename, basedir)
            fin = fin + '.ps'
        else:
            #conversion failed
            fout = None

    if fin.endswith('.ps'):
        fout = fin.replace('.ps','.png')
    #convert to png
        if pyimage:
            f = Image(fin)
            f.rotate(90)
            f.write(fout)
        else:
            cmd = ['convert','-rotate','90',fin,'png:%s' % fout]
            subprocess.call(cmd, shell=False,
                            stdout=open('/dev/null','w'))

    return fout


def load_data(fname):
    """
    read data stored in a simple txt file with (at least) one column:
    filename 

    subsequent columns are added based on a users votes

    Notes:
    We also expect the first row to be a '#' comment line with the 
    labels for each column. This must be, at least, '#fname '

    We will create the AI row if necessary and a 'dummy' voter row

    """
    print "Opening %s" % fname
    if fname.endswith('.npy'):
        data = np.load(fname)
    else:
        f = open(fname,'r')
        l1 = f.readline()
        f.close()
        if '#' not in l1:
            print "Expected first line to be a comment line describing the columns"
            print "format: fname voter1 voter2 ..."
            print "We assume voter1 = AI"
            ncol = len(l1.split())
            if ncol > 1:
                colnames = ['fname','AI']
                coltypes = ['|S130', 'f8']
                for n in range(ncol-2):
                    colnames.append('f%s' % n)
                    coltypes.append('f8')
            else:
                colnames = ['fname']
                coltypes = ['|S130']
        else:
            l1 = l1.strip('#')
            colnames = l1.split()
            #fname, AI
            coltypes = ['|S130','f8']
            for n in range(len(colnames)-2):
                coltypes.append('f8')

        data = np.recfromtxt(fname, dtype={'names':colnames,'formats':coltypes})

# add in a dummy AI vote if it doesn't already exist
    if 'AI' not in data.dtype.names:
        data = add_voter('AI', data)
    if len(data.dtype.names) < 3: #fname, AI
        name = inputbox('Voter chooser',\
                            'No user voters found in %s. Add your voting name' % fname)
        data = add_voter(name, data)

    return data


def add_voter(voter, data):
    """
    add a field 'voter' to the data array

    """
    if voter not in data.dtype.names:
        nrow = len(data)
        if voter == 'AI':
            this_dtype = 'f8'
        else:
            this_dtype = 'f8'
        nvote = np.zeros(nrow,dtype=this_dtype)*np.nan
        dtype = data.dtype.descr
        if voter == 'AI':
            dtype.append((voter,'f8'))
        else:
            dtype.append((voter,'f8'))
        newdata = np.zeros(nrow,dtype=dtype)
        newdata[voter] = nvote
        for name in data.dtype.names:
            newdata[name] = data[name]
        data = newdata.view(np.recarray)
    return data


def inputbox(title='Input Box', label='Please input the value',
        parent=None, text=''):
    """
    dialog with a input entry
    
    return text , or None
    """

    dlg = Gtk.Dialog(title, parent, Gtk.DialogFlags.DESTROY_WITH_PARENT,
            (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OK, Gtk.ResponseType.OK    ))
    lbl = Gtk.Label(label)
    lbl.set_alignment(0, 0.5)
    lbl.show()
    dlg.vbox.pack_start(lbl, False, False, 0)
    entry = Gtk.Entry()
    if text: entry.set_text(text)
    entry.show()
    dlg.vbox.pack_start(entry, False, True, 0)
    dlg.set_default_response(Gtk.ResponseType.OK)
    resp = dlg.run()
    text = entry.get_text()
    dlg.hide()
    if resp == Gtk.ResponseType.CANCEL:
        return None
    return text


def messagedialog(dialog_type, short, long=None, parent=None,
                buttons=Gtk.ButtonsType.OK, additional_buttons=None):
    d = Gtk.MessageDialog(parent=parent, flags=Gtk.DialogFlags.MODAL,
                        type=dialog_type, buttons=buttons)
    
    if additional_buttons:
        d.add_buttons(*additional_buttons)
    
    d.set_markup(short)
    
    if long:
        if isinstance(long, Gtk.Widget):
            widget = long
        elif isinstance(long, basestring):
            widget = Gtk.Label()
            widget.set_markup(long)
        else:
            raise TypeError("long must be a Gtk.Widget or a string")
        
        expander = Gtk.Expander(_("Click here for details"))
        expander.set_border_width(6)
        expander.add(widget)
        d.vbox.pack_end(expander)
        
    d.show_all()
    response = d.run()
    d.destroy()
    return response


if __name__ == '__main__':        
    
    #did we pass an input file:
    args = sys.argv[1:]
    data = None
    if len(args) > 0:
        data = args[0]#load_data(args[0])

    app = MainFrameGTK(data=data)    
    Gtk.main()
