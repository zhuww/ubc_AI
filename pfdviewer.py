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
import cPickle
import fractions
import glob
import numpy as np
import os
import shutil
import subprocess
import sys
import tempfile
from os.path import abspath, basename, dirname, exists

from gi.repository import Gtk, Gdk
import pylab as plt

#next taken from ubc_AI.training and ubc_AI.samples
from training import pfddata
from sklearn.decomposition import RandomizedPCA as PCA
import known_pulsars as KP

#check for ps --> png conversion utilities
try:
    from PythonMagick import Image
    pyimage = True
except ImportError:
    pyimage = False
if not pyimage:
    conv = subprocess.call(['which','convert'], stdout=open('/dev/null','w'))
    if conv == 1:
        print "pfdviewer requires imagemagik's convert utility"
        print "or, PythonMagick. Exiting..."
        sys.exit()

#PRESTO's show_pfd command:
show_pfd = False
for p in os.environ.get('PATH').split(':'):
    cmd = '%s/show_pfd' % p
    if exists(cmd):
        show_pfd = cmd
        break
if not show_pfd:
    print "\tCouldn't find PRESTO's show_pfd executable"
    print "\t This will limit functionality"
try:
    from prepfold import pfd as PFD
except(ImportError):
    print "please add PRESTO's modules to your python path for more functionality"
    PFD = None

#do not allow active voter = sort column. warn once
have_warned = False


#iter on each "n". auto-save after every 10
cand_vote = 0
#store AI_view png's in a temporary dir
tempdir = tempfile.mkdtemp(prefix='AIview_')
atexit.register(lambda: shutil.rmtree(tempdir))
bdir = '/'.join(__file__.split('/')[:-1])


class MainFrameGTK(Gtk.Window):
    """This is the Main Frame for the GTK application"""
    
    def __init__(self, data=None):
        Gtk.Window.__init__(self, title='pfd viewer')
        if bdir:
            self.gladefile = "%s/pfdviewer.glade" % bdir
        else:
            self.gladefile = "pfdviewer.glade"
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
        self.pmatchwin_tog = self.builder.get_object('pmatchwin_tog')
        self.col_options = []
        self.col1 = self.builder.get_object('col1')
        self.col2 = self.builder.get_object('col2')
        self.active_col1 = None
        self.active_col2 = None
        self.view_limit = self.builder.get_object('view_limit')
        self.limit_toggle = self.builder.get_object('limit_toggle')
        self.verbose_match = self.builder.get_object('verbose_match')
        self.matchsep = self.builder.get_object('matchsep')
        self.advance_next = self.builder.get_object('advance_next')
        self.advance_col = self.builder.get_object('advance_col')

        #tmpAI window 
        self.tmpAI_win = self.builder.get_object('tmpAI_votemat')
        self.tmpAI_overall = self.builder.get_object('overall_vote')
        self.tmpAI_phasebins = self.builder.get_object('phasebins_vote')
        self.tmpAI_intervals = self.builder.get_object('intervals_vote')
        self.tmpAI_subbands = self.builder.get_object('subbands_vote')
        self.tmpAI_DMbins = self.builder.get_object('DMbins_vote')
        self.tmpAI_tog = self.builder.get_object('tmpAI_tog')
        self.tmpAI_lab = self.builder.get_object('tmpAI_lab')
        self.tmpAI = None 
        
        self.pmatch_win = self.builder.get_object('pmatch_win')
        self.pmatch_tree = self.builder.get_object('pmatch_tree')
        self.pmatch_store = self.builder.get_object('pmatch_store')
        self.pmatch_lab = self.builder.get_object('pmatch_lab')
        self.pmatch_tree.connect("cursor-changed", self.on_pmatch_select_row)
        self.pmatch_tree.connect("row-activated", self.on_pmatch_row_activated)
        self.pmatch_tree.hide()
        self.pmatch_lab.hide()
#allow Ctrl+s like key-functions        
        self.modifier = None
#where are pfd/png/ps files stored
        self.basedir = '.'

#define the list columns
        for vi, v  in enumerate(['n','fname','col1_prob', 'col2_prob']):
            #only show two columns at a time
            cell = Gtk.CellRendererText()
            col = Gtk.TreeViewColumn(v, cell, text=vi)
            col.set_property("alignment", 0.5)
            col.set_sort_indicator(True)
            col.set_sort_column_id(vi)
            if v == 'fname':
                expcol = col
                col.set_expand(True)
                col.set_max_width(180)
            elif v == 'n':
                col.set_expand(False)
                col.set_max_width(42)
                col.set_sort_indicator(False)
            else:
                col.set_expand(False)
                col.set_max_width(80)
            self.pfdtree.append_column(col)
        self.pfdtree.set_expander_column(expcol)
        self.pfdstore.set_sort_column_id(2,1)#arg1=col, arg2=sort/revsort

# set up the matching-pulsar tree
        self.pmatch_tree_init()

        ## data-analysis related objects
        self.voters = []
        self.savefile = None
        self.loadfile = None 
        self.knownpulsars = {}
        #ATNF, PALFA and GBNCC list of known pulsars
        if exists('%s/known_pulsars.pkl' % bdir):
            self.knownpulsars = cPickle.load(open('%s/known_pulsars.pkl' % bdir))
        elif exists('known_pulsars.pkl'):
            self.knownpulsars = cPickle.load(open('known_pulsars.pkl'))
        else:
            self.knownpulsars = KP.get_allpulsars()
        #if we were passed a data file, read it in
        if data != None:
            self.on_open(event='load', fin=data)
        else:
            self.data = None
        # start with default and '<new>' voters
        if self.data != None:
            self.voters = list(self.data.dtype.names[1:]) #strip off 'fname'
            for v in self.voters:
                if v not in self.col_options:
                    self.col_options.append(v)
                    self.col1.append_text(v)
                    self.col2.append_text(v)
    #<new> is a special case used to add new voters
            if '<new>' not in self.voters:
                self.voters.insert(0,'<new>')

            #display first voter column from input data, making it "active"
            self.active_col1 = self.col_options.index(self.voters[1])
            self.col1.set_active(self.active_col1)
            #make active vote the first non-"AI" column
            av = np.where(np.array(self.data.dtype.names[1:] != 'AI'))[0]
            if len(av) == 0:
                self.active_voter = 1
                self.statusbar.push(0, 'Warning, voting overwrites AI votes')
            else:
                idx = self.voters.index(self.data.dtype.names[1:][av[0]]) 
                if 'AI' in self.voters: 
                    idx += 1
                self.active_voter = idx 
                self.voterbox.set_active(idx)

#set up the second column if there are multiple "voters"
            if len(self.voters) > 2:
#                self.active_voter = 1
#                self.voterbox.set_active(self.active_voter)
                if 'AI' in self.voters:
                    self.active_col2 = self.col_options.index(self.voters[self.active_voter]) 
                    self.col2.set_active(self.active_col2)
                else:
                    self.active_col2 = self.col_options.index(self.voters[self.active_voter+1]) 
                    self.col2.set_active(self.active_col2)

            self.dataload_update()
        #put cursor on first col. if there is data
            self.pfdtree.set_cursor(0)

        else:
            self.statusbar.push(0,'Please load a data file')
            self.voters = []
            self.active_voter = None

#keep track of the AI view files created (so we don't need to generate them)
        self.AIviewfiles = {}


############################
## data-manipulation actions
    def on_sep_change(self, widget):
        """
        respond to changes in the pulsar matching separation
        """
        self.find_matches()

    def on_viewlimit_toggled(self, widget):
        self.on_view_limit_changed( widget)

    def on_verbose_match(self, widget):
        self.find_matches()
    def on_view_limit_changed(self, widget):
        """
        change what is displayed when the view limit is changed

        """
        col1 = self.col1.get_active_text()
        col2 = self.col2.get_active_text()
        idx1 = self.col_options.index(col1)
        if col2 == None:
            col2 = col1
            self.col2.set_active(idx1)
        idx2 = self.col_options.index(col2)
        limtog = self.limit_toggle.get_active()
        lim = self.view_limit.get_value()

#turn off the model first for speed-up
        self.pfdtree.set_model(None)
        self.pfdstore.clear()

        if idx1 != idx2:
            data = self.data[['fname',col1,col2]]
            data.sort(order=[col1,'fname'])
            limidx = data[col1] >= lim - 1e-5
            if np.any(limidx) and limtog:
                data = data[limidx]
                self.statusbar.push(0,'Showing %s/%s candidates above %s' %
                                    (limidx.sum(),len(limidx),lim))
            else:
                self.statusbar.push(0,'No %s candidates > %s. Showing all' % (col1, lim))
            for vi, v in enumerate(data[::-1]):
                d = (vi,) + v.tolist()
                self.pfdstore.append(d)
        else:
            data = self.data[['fname',col1]]
            data.sort(order=[col1,'fname'])
            limidx = data[col1] >= lim - 1e-5
            if np.any(limidx) and limtog:
                data = data[limidx]
                self.statusbar.push(0,'Showing %s/%s candidates above %s' %
                                    (limidx.sum(),len(limidx),lim))
            else:
                self.statusbar.push(0,'No %s candidates > %s. Showing all' % (col1, lim))
            for vi, v in enumerate(data[::-1]):
                v0, v1 = v
                self.pfdstore.append((vi,v0,v1,v1))
                    
        self.pfdtree.set_model(self.pfdstore)
        self.find_matches()

    def on_pfdwin_key_press_event(self, widget, event):
        """
        controls keypresses on over-all window

        """
        global cand_vote, have_warned
        
#key codes which change the voting data
        votes = {'1':1., 'p':1.,    #pulsar
                 'r': np.nan,      #reset to np.nan
                 '5':.5, 'm':.5,   #50/50 pulsar/rfi
                 'k':2.,           #known pulsar
                 'h':3.,           #harmonic of known
                 '0':0.            #rfi (not a pulsar)
                 }
                 

        key = Gdk.keyval_name(event.keyval)
        ctrl = event.state &\
            Gdk.ModifierType.CONTROL_MASK
        if self.active_voter:
            act_name = self.voters[self.active_voter]
        else:
            act_name = 'AI'

            
#don't allow voting/actions if sort_column = active voter
        col1 = self.col1.get_active_text()
        col2 = self.col2.get_active_text()
        sort_id = self.pfdstore.get_sort_column_id()[0]
        #0=number, 1=fname, 2=col1, 3=col2
        if (sort_id == 2) and (act_name == col1) and key in votes:
            note = "Note. Voting disabled when active voter = sort column"
            if not have_warned:
                dlg = Gtk.MessageDialog(self, 0, Gtk.MessageType.INFO,
                                        Gtk.ButtonsType.OK, note)
                response = dlg.run()
                dlg.destroy()
                have_warned = True
                self.statusbar.push(0, 'Vote not recorded. voter = sort column')
            else:
                self.statusbar.push(0, note)
            return
        elif (sort_id == 3) and (act_name == col2) and key in votes:
            note = "Note. Voting disabled when active voter = sort column"
            if not have_warned:
                dlg = Gtk.MessageDialog(self, 0, Gtk.MessageType.INFO,
                                        Gtk.ButtonsType.OK,note)
                response = dlg.run()
                dlg.destroy()
                have_warned = True
                self.statusbar.push(0, 'Vote not recorded. voter = sort column')
            else:
                self.statusbar.push(0, note)
            return
        elif act_name == 'AI' and key  in votes:
            note = 'Note: AI voter is not editable. Change active voter'
            print note
            self.statusbar.push(0, note)
            return 


        (model, pathlist) = self.pfdtree.get_selection().get_selected_rows()
#only use first selected object
        if len(pathlist) == 0:
            #nothing selected, so go back to first
            path = None
            this_iter = None
            next_path = None
        else:
            path = pathlist[0]
            this_iter = model.get_iter(path)
            next_iter = model.iter_next(this_iter)
            advance = self.advance_next.get_active()
            adv_col = int(self.advance_col.get_value()) + 1 #plus 'number' and 'fname'
            if advance:
                data = self.pfdstore[next_iter]
                while not np.isnan(data[adv_col]):
                    next_iter = model.iter_next(next_iter)
                    if next_iter == None:
                        self.statusbar.push(0,'No unranked candidates in voter col %i. You are Done!' % adv_col)

                        break
                    else:
                        data = self.pfdstore[next_iter]
        #keep modifier keys until they are released
        if key in ['Control_L','Control_R','Alt_L','Alt_R']:
            self.modifier = key

        if key == 'q' and self.modifier in\
                ['Control', 'Control_L', 'Control_R', 'Primary']:
            self.on_menubar_delete_event(widget, event)

        elif key == 's' and self.modifier in ['Control_L', 'Control_R', 'Primary']:
            self.on_save(widget)
            
        elif key == 'l' and self.modifier in ['Control_L', 'Control_R', 'Primary']:
            self.on_open()

        elif key == 'n':           
            self.pfdtree_next(model, next_iter)
        elif key == 'b':
            self.pfdtree_prev()

        #data-related (needs to be loaded)
        if self.data != None:

            if key in votes:
                value = votes[key]
                self.pfdtree_next(model, next_iter)
                fname = self.pfdstore_set_value(value, this_iter=this_iter, \
                                                    return_fname=True)
                
                if key in ['1', 'p', 'm', '5', 'h']:
                    self.add_candidate_to_knownpulsars(fname)

            elif key == 'c':
                # cycle between ranked candidates
                self.pmatchtree_next()


            cand_vote += 1
            if cand_vote//10 == 1:
                if self.autosave.get_active():
                    self.on_save()
                else:
                    self.statusbar.push(0,'Remember to save your output')
                r = cand_vote // 30
                d = cand_vote % (r*30)
                if d == 0:
                    note = 'Congratulations  %s. You are a level %s classifier\n' % (act_name, r)
                    note += 'Only %s votes to the next level, and possibly a free coffee from Aaron and Weiwei' % ((r+1)*30)
                    
                    dlg = Gtk.MessageDialog(self, 0, Gtk.MessageType.INFO,
                                        Gtk.ButtonsType.OK, note)
                    response = dlg.run()
                    dlg.destroy()

    def add_candidate_to_knownpulsars(self, fname):
        """
        as a user ranks candidates, add the pulsar candidates to the list
        of known pulsars

        input: 
        filename of the pfd file

        """
        
        if exists(fname) and fname.endswith('.pfd'):
            pfd = pfddata(fname)
            pfd.dedisperse()
            dm = pfd.bestdm
            ra = pfd.rastr
            dec = pfd.decstr
            p0 = pfd.bary_p1
            name = basename(fname)
            if float(pfd.decstr.split(':')[0]) > 0:
                sgn = '+'
            else:
                sgn = ''
                name = 'J%s%s%s' % (''.join(pfd.rastr.split(':')[:2]), sgn,\
                                        ''.join(pfd.decstr.split(':')[:2]))
#            this_pulsar = KP.pulsar(fname, name, ra, dec, p0*1e-3, dm)
            this_pulsar = KP.pulsar(fname, name, ra, dec, p0, dm, catalog='local')
                
            self.knownpulsars[fname] = this_pulsar


    def dataload_update(self):
        """
        update the pfdstore whenever we load in a new data file
        
        set columns to "fname, AI, [1st non-AI voter... if exists]"

        """
        if self.active_voter:
            act_name = self.voters[self.active_voter]
#update the Treeview... 
            if self.data != None:

                col1 = self.col1.get_active_text()
                col2 = self.col2.get_active_text()
                idx1 = self.col_options.index(col1)
                if col2 == None:
                    col2 = col1
                    self.col2.set_active(idx1)
                idx2 = self.col_options.index(col2)

                
                limtog = self.limit_toggle.get_active()
                lim = self.view_limit.get_value()

#turn off the model first for speed-up
                self.pfdtree.set_model(None)
                self.pfdstore.clear()

                self.active_col1 = idx1
                self.active_col2 = idx2
                if idx1 != idx2:
                    data = self.data[['fname',col1,col2]]
                    data.sort(order=[col1,'fname'])
                    limidx = data[col1] >= lim - 1e-5
                    if np.any(limidx) and limtog:
                        data = data[limidx]
                        self.statusbar.push(0,'Showing %s/%s candidates above %s' %
                                            (limidx.sum(),len(limidx),lim))
                    else:
                        self.statusbar.push(0,'No %s candidates > %s. Showing All.' % (col1, lim))
                    for vi, v in enumerate(data[::-1]):
                        d = (vi,) + v.tolist()
                        self.pfdstore.append(d)
                else:
                    data = self.data[['fname',col1]]
                    data.sort(order=[col1,'fname'])
                    limidx = data[col1] >= lim - 1e-5
                    if np.any(limidx) and limtog:
                        data = data[limidx]
                        self.statusbar.push(0,'Showing %s/%s candidates above %s' %
                                            (limidx.sum(),len(limidx),lim))
                    else:
                        self.statusbar.push(0,'No %s candidates > %s. Showing all.' % (col1, lim))
                    for vi, v in enumerate(data[::-1]):
                        v0, v1 = v
                        self.pfdstore.append((vi,v0,v1,v1))
                        
                self.pfdtree.set_model(self.pfdstore)
                self.find_matches()

    def on_col_changed(self, widget):
        """
        change the pfdstore whenever we change the column box

        """
        if self.data != None and self.active_col1 != None and self.active_col2 != None:

            col1 = self.col1.get_active_text()
            col2 = self.col2.get_active_text()
            idx1 = self.col_options.index(col1)
            if col2 == None:
                col2 = col1
                self.col2.set_active(idx1)
            idx2 = self.col_options.index(col2)

            if (self.active_col1 != idx1) or (self.active_col2 != idx2):
                self.pfdtree.set_model(None)
                self.pfdstore.clear()
                self.active_col1 = idx1
                self.active_col2 = idx2
                
                limtog = self.limit_toggle.get_active()
                lim = self.view_limit.get_value()
                if idx1 != idx2:
                    data = self.data[['fname',col1,col2]]
                    data.sort(order=[col1,'fname'])
                    limidx = data[col1] >= lim - 1e-5
                    if np.any(limidx) and limtog:
                        data = data[limidx]
                        self.statusbar.push(0,'Showing %s/%s candidates above %s' %
                                             (limidx.sum(),len(limidx),lim))
                    else:
                        self.statusbar.push(0,'No %s candidates > %s' % (col1, lim))
                    for vi, v in enumerate(data[::-1]):
                        d = (vi,) + v.tolist()
                        self.pfdstore.append(d)
                else:
                    data = self.data[['fname',col1]]
                    data.sort(order=[col1,'fname'])
                    limidx = data[col1] >= lim - 1e-5
                    if np.any(limidx) and limtog:
                        data = data[limidx]
                        self.statusbar.push(0,'Showing %s/%s candidates above %s' %
                                            (limidx.sum(),len(limidx),lim))
                    else:
                        self.statusbar.push(0,'No %s candidates > %s' % (col1, lim))
                    for vi, v in enumerate(data[::-1]):
                        v0, v1 = v
                        self.pfdstore.append((vi, v0, v1, v1))

                self.pfdtree.set_model(self.pfdstore)
                self.find_matches()

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
            fname = os.path.join(self.basedir, tmpstore.get_value(tmpiter, 1))

#are we displaying the prediction from a tmpAI?            
            if exists(fname) and fname.endswith('.pfd') and (self.tmpAI != None) and self.tmpAI_tog.get_active():
                pfd = pfddata(fname)
                pfd.dedisperse()
                avgs = feature_predict(self.tmpAI, pfd)
                self.update_tmpAI_votemat(avgs)
                disp_apnd = '(tmpAI: %0.3f)' % (self.tmpAI.predict_proba(pfd)[...,1][0])
            else:
                disp_apnd = ''

# find/create png file from input file
            fpng = self.create_png(fname)

            #update the basedir if necessary 
            if not exists(fpng):
                fname = self.find_file(fname)
                fpng = self.create_png(fname)
            
            #we are not doing "AI view" of data
            if not self.aiview.get_active():
                if fpng and exists(fpng):
                    self.image.set_from_file(fpng)
                    self.image_disp.set_text('displaying : %s %s' % 
                                             (basename(fname), disp_apnd))
                else:
                    note = "Failed to generate png file %s" % fname
                    print note
                    self.statusbar.push(0,note)
                    self.image.set_from_file('')

            else:

                #we are doing the AI view of the data
                fpng= ''
                if exists(fname) and fname.endswith('.pfd'):
                    self.statusbar.push(0,'Generating AIview...')

                    #have we generated this AIview before?
                    fpng = self.check_AIviewfile_match(fname)
                    if not fpng:
                        fpng = self.generate_AIviewfile(fname)
                    if fpng and exists(fpng):
                        self.image.set_from_file(fpng)
                        self.image_disp.set_text('displaying : %s %s' % (fname, disp_apnd))
                    else:
                        note = "Failed to generate png file %s" % fname
                        print note
                        self.statusbar.push(0,note)
                        self.image.set_from_file('')
                        self.image_disp.set_text('displaying : %s %s' % (fname, disp_apnd))
                elif fname.endswith('.png'):
                    note = "Can't generate AIview from png files"
                    self.statusbar.push(0, note)
                    fpng = fname
                    self.image.set_from_file(fpng)
                    self.image_disp.set_text('displaying: %s %s' % (fpng, disp_apnd))
            self.find_matches()

    def create_png(self, fname):
        """
        given some pfd or ps file, create the png file
        if it doesn't already exist

        """

        if fname.endswith('.ps'):
            fpng = fname.replace('.ps', '.png')
            if not exists(fpng):
                #convert ps file to png
                if exists(fname):
                    fpng = convert(fname)
                #convert using the pfd file
                else:
                    pfdfile = os.path.splitext(fname)[0]
                    if exists(pfdfile):
                        fname = pfdfile
                        fpng = convert(fname)
        elif fname.endswith('.pfd'):
            fpng = '%s.png' % fname
            if not exists(fpng):
                fpng = convert(fname)
        elif fname.endswith('.png') or fname.endswith('.jpg'):
            #convert from ps or pfd if we can't find the png
            if not exists(fname):
                psfile = fname.replace('.png','.ps')
                if not exists(psfile):
                    pfdfile = os.path.splitext(fname)[0]
                    if exists(pfdfile):
                        fname = convert(pfdfile)
                else:
                    fname = convert(psfile)
            fpng = fname
        else:
            note = "Don't recognize file %s" % fname
            print note
            self.statusbar.push(0, note)
            fpng = ''


    #see if png exists locally already, otherwise generate it

        if not exists(fpng) and exists(fname):
            #convert to png (convert accepts .ps, or .pfd file)
            fpng = convert(fname)

        return fpng

    def find_file(self, fname):
        """
        make sure we can find the file
        
        return:
        png file name if we find it
        '' empty string otherwise

        """
        if not exists(fname):
            print "Can't find %s" % fname
            dlg = Gtk.MessageDialog(self, 0, Gtk.MessageType.INFO,
                                    Gtk.ButtonsType.OK,
                                    "Can't find %s.\n\n Please select a base path for this file" % fname)
            response = dlg.run()
            dlg.destroy()
            dialog = Gtk.FileChooserDialog("Choose base path for %s" % \
                                               os.path.splitext(fname)[0],
                                           self, Gtk.FileChooserAction.SELECT_FOLDER,
                                           (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                                            "Select", Gtk.ResponseType.OK))
            dialog.set_default_size(800,400)
            response = dialog.run()
            if response == Gtk.ResponseType.OK:
                self.basedir = dialog.get_filename()
                print "Setting basepath to %s" % self.basedir
                self.statusbar.push(0, 'Setting basepath to %s' % self.basedir)
            else:
                self.basedir = './'
            dialog.destroy()
            fname = basename(fname)
            fname = os.path.join(abspath(self.basedir), fname)
            if exists(fname):
                return fname
            else:
                return basename(fname)
        else:
            return abspath(fname)
        
    def generate_AIviewfile(self, fname):
        """
        Given some PFD file and the AI_view flag, generate the png
        file for this particular view. 

        We save the files in /tmp/AIview*, 

        """
        
        pfd = pfddata(fname)
        plt.figure(figsize=(8,5.9))
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
#            ax = plt.subplot(2, 2, subplt+1)
            if subplt == 0:
                ax = plt.subplot2grid((3,2),(0,0))
                data = pfd.getdata(phasebins=nbins)
                if npca_comp:
                    pca = PCA(n_components=npca_comp)
                    pca.fit(data)
                    pcadata = pca.transform(data)
                    data = pca.inverse_transform(pcadata)
                ax.plot(data)
                ax.set_title('pulse profile (bins, pca) = (%s,%s)'%(nbins,npca_comp))
            elif subplt == 1:
                ax = plt.subplot2grid((3,2), (0,1), rowspan=2)
                data = pfd.getdata(subbands=nbins)
                if npca_comp:
                    #note: PCA is best set when fed many samples, not one
                    pca = PCA(n_components=npca_comp)
                    rd = data.reshape(nbins,nbins)
                    pca.fit(rd)
                    data = pca.inverse_transform(pca.transform(rd)).flatten()
                ax.imshow(data.reshape(nbins, nbins),
                          cmap=plt.cm.gray)
                ax.set_title('subbands (bins, pca) = (%s,%s)'%(nbins,npca_comp))
            elif subplt == 2:
                ax = plt.subplot2grid((3,2), (1,0), rowspan=2)
                data = pfd.getdata(intervals=nbins)
                if npca_comp:
                    #note: PCA is best set when fed many samples, not one
                    pca = PCA(n_components=npca_comp)
                    rd = data.reshape(nbins,nbins)
                    pca.fit(rd)
                    data = pca.inverse_transform(pca.transform(rd)).flatten()
                ax.imshow(data.reshape(nbins,nbins),
                                cmap=plt.cm.gray)
                ax.set_title('intervals (bins, pca) = (%s,%s)'%(nbins,npca_comp))
            elif subplt == 3:
                ax = plt.subplot2grid((3,2), (2,1))
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

        
    def pfdstore_set_value(self, value, this_iter=None, return_fname=False):
        """
        update the pfdstore value for the given path
        
        Args:
        Value: the voter prob (ranking) to assign
        return_fname: return the filename of the pfd file
        """
        if this_iter != None:

#update self.data (since dealing with TreeStore blows my mind)
            n, fname, x, oval = self.pfdstore[this_iter]
            col1 = self.col1.get_active_text()
            col2 = self.col2.get_active_text()

            idx = self.data['fname'] == fname
            if self.active_voter:
                act_name = act_name = self.voters[self.active_voter]
                self.data[act_name][idx] = value
                if col1 == act_name:
                    self.pfdstore[this_iter][2] = value
                if col2 == act_name:
                    self.pfdstore[this_iter][3] = value

            if return_fname:
                return fname
            else:
                return None

    def on_pmatch_row_activated(self, widget, event, data=None):
        """
        respond to double-clicks in pmatch tree
        if it is a candidate file, we move the pfdtree to this row

        """
        #get name of double-clicked object
        (model, pathlist) = self.pmatch_tree.get_selection().get_selected_rows()
        if len(pathlist) != 0:
            path = pathlist[0]
            match_iter = model.get_iter(path)
            fname = self.pmatch_store[match_iter][0]

            
        # scroll through entire list of candidates until
        # we find the candidate, then switch to it
        pfd_iter = self.pfdstore.get_iter_first()
        found = False
        while pfd_iter is not None:
            candname = self.pfdstore[pfd_iter][1]
            if basename(candname) == basename(fname):
                found = True
                break
            pfd_iter = self.pfdstore.iter_next(pfd_iter)
        
        if found:
            (pfdtreemodel, pdel) = self.pfdtree.get_selection().get_selected_rows()
            path = pfdtreemodel.get_path(pfd_iter)
            self.pfdtree.set_cursor(path)
            self.pfdtree.scroll_to_cell(path)
            pass
    def on_pmatch_select_row(self, event=None):
        """
        cycle/display the list of candidate matches
        
        """
# get name of "selected" candidate
        gsel = self.pfdtree.get_selection()
        if gsel:
            tmpstore, tmpiter = gsel.get_selected()
        else:
            tmpiter = None

        ncol = self.pfdstore.get_n_columns()
        if tmpiter != None:
#            candname = os.path.join(self.basedir, tmpstore.get_value(tmpiter, 0))
            candname = tmpstore.get_value(tmpiter, 1)

            (model, pathlist) = self.pmatch_tree.get_selection().get_selected_rows()
    #only use first selected object
            if len(pathlist) > 0:
                #nothing selected, so go back to first
                path = pathlist[0]
                tree_iter = model.get_iter(path)
    #update self.data (since dealing with TreeStore blows my mind)
                fname, p0, dm, ra, dec, vote = self.pmatch_store[tree_iter]
                if self.knownpulsars.has_key(fname):
                    cat = self.knownpulsars[fname].catalog
                    self.statusbar.push(0,'Selected match found: %s' % cat)
                else:
                    z = [v.catalog for v in self.knownpulsars.values() if v.name == fname]
                    if len(z) == 1:
                        self.statusbar.push(0, 'Selected match found: %s' % z[0])
                    else:
                        self.statusbar.push(0, 'Selected match found: local')
                if fname.endswith('.pfd'):
                    fname = os.path.join(self.basedir, fname)
                    if not self.aiview.get_active():
                        # find/create png file from input file
                        fpng = self.create_png(fname)
                       #update the basedir if necessary 
                        if not exists(fpng):
                            fname = self.find_file(fname)
                            fpng = self.create_png(fname)
                    elif exists(fname):
                        fpng = self.check_AIviewfile_match(fname)
                        if not fpng:
                            fpng = self.generate_AIviewfile(fname)
                    else:
                        fpng = ''

#are we displaying the prediction from a tmpAI?            
                    if exists(fname) and fname.endswith('.pfd') and (self.tmpAI != None) and self.tmpAI_tog.get_active():
                        pfd = pfddata(fname)
                        pfd.dedisperse()
                        avgs = feature_predict(self.tmpAI, pfd)
                        self.update_tmpAI_votemat(avgs)
                        disp_apnd = '(tmpAI: %0.3f)' % (self.tmpAI.predict_proba(pfd)[...,1][0])
                    else:
                        disp_apnd = ''
                        
    #                print "Showing image",fname
                    if fpng and exists(fpng):
                        self.image.set_from_file(fpng)
                        self.image_disp.set_text('displaying : %s %s' % \
                                                     (basename(fname), disp_apnd))
                        if basename(fname) == basename(candname):
                            disp = 'Possible matches to %s' % basename(candname)
                        else:
                            idx = self.data['fname'] == candname
                            if len(idx[idx]) > 0:
                                vote = self.data[self.voters[self.active_voter]][idx][0]
                            else:
                                vote = np.nan
                            disp = 'Possible matches to %s\n' % basename(candname)
                            disp += '               (displaying %s (%s))' % (basename(fname), vote)
                        self.pmatch_lab.set_text(disp)

    def on_tmpAI_toggled(self, event):
        """
        if 'on', load a pickled classifier and display
        its' prediction. 

        Otherwise 'forget' the loaded classifier

        """
        if self.tmpAI_tog.get_active():
            dialog = Gtk.FileChooserDialog("load a pickled classifier", self,
                                           Gtk.FileChooserAction.OPEN,
                                           (Gtk.STOCK_CANCEL,
                                            Gtk.ResponseType.CANCEL,
                                            Gtk.STOCK_OPEN, Gtk.ResponseType.OK))
            response = dialog.run()
            if response == Gtk.ResponseType.OK:
                fname = dialog.get_filename()
                try:
                    self.tmpAI = cPickle.load(open(fname))
                    self.tmpAI_lab.set_text(basename(fname))
                except (IOError, EOFError, AttributeError):
                    print "couldn't load %s" % fname
                    self.statusbar.push(0,"couldn't load %s" % fname)
                    self.tmpAI = None
                    self.tmpAI_lab.set_text('')
            else:
                self.tmpAI = None
                self.tmpAI_lab.set_text('')
                self.tmpAI_tog.set_active(0)
            dialog.destroy()
        else:
            self.tmpAI = None
        
        if self.tmpAI == None:
            self.tmpAI_win.hide()
        else:
            self.tmpAI_win.show_all()

    def update_tmpAI_votemat(self, avgs):
        """
        given a dictionary of performances for the various features,
        update the tmpAI_vote window

        """

        if 'phasebins' in avgs:
            self.tmpAI_phasebins.set_text('profile : %0.3f' % avgs['phasebins'])
        else:
            self.tmpAI_phasebins.set_text('profile : N/A')
        if 'intervals' in avgs:
            self.tmpAI_intervals.set_text('intervals : %0.3f' % avgs['intervals'])
        else:
            self.tmpAI_intervals.set_text('intervals : N/A')
        if 'subbands' in avgs:
            self.tmpAI_subbands.set_text('subbands : %0.3f' % avgs['subbands'])
        else:
            self.tmpAI_subbands.set_text('subbands : N/A')
        if 'DMbins' in avgs:
            self.tmpAI_DMbins.set_text('DMbins : %0.3f' % avgs['DMbins'])
        else:
            self.tmpAI_DMbins.set_text('DMbins : N/A')
        
        if 'overall' in avgs:
            self.tmpAI_overall.set_text('overall voting performance: %0.4f' % avgs['overall'])
        else:
            self.tmpAI_overall.set_text('overall voting performance: N/A')

                

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
                    note = "adding voter data for %s" % d
                    print note
                    self.statusbar.push(0, note)
                    self.voters.append(d)
                    self.data = add_voter(d, self.data)
                    self.voterbox.append_text(d)
                    if d not in self.col_options:
                        self.col1.append_text(d)
                        self.col2.append_text(d)
                        self.col_options.append(d)
                    self.active_voter = len(self.voters) - 1
                else:
                    note = 'User already exists. switching to it'
                    print note
                    self.statusbar.push(0, note)
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
            if self.savefile == None and cand_vote > 0:
                savdlg = Gtk.MessageDialog(self, 0, Gtk.MessageType.INFO,
                                           Gtk.ButtonsType.OK_CANCEL,
                                           "Save your votes?")
                savrep = savdlg.run()
                if savrep == Gtk.ResponseType.OK:
                    self.on_save()
                savdlg.destroy()

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
        
        Also, check all the user's votes and add the candidates to self.known_pulsars

        """

        fname = ''
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
#                print "File selected: " + fname
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
            for v in self.voters:
                if v not in self.col_options:
                    self.col_options.append(v)
                    self.col1.append_text(v)
                    self.col2.append_text(v)

#<new> is a special case used to add new voters
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
            if self.active_col1:
                self.col1.set_active(self.active_col1)
            else:
                self.col1.set_active(0)
                self.active_col1 = 0
            if self.active_col2:
                self.col2.set_active(self.active_col2)
            else:
                self.col2.set_active(1)
                self.active_col2 = 1
            self.statusbar.push(0,'Loaded %s candidates' % len(self.data))
        self.dataload_update()
        self.pfdtree.set_cursor(0)

        if self.knownpulsars == None:
            self.statusbar.push(0,'Downloading ATNF, PALFA and GBNCC list of known pulsars')
            self.knownpulsars = KP.get_allpulsars()
            self.statusbar.push(0,'Downloaded %s known pulsars for x-ref'\
                                % len(self.knownpulsars))

#add all the candidates ranked as pulsars to the list of known_pulsars
        for v in self.data.dtype.names[1:]:
            #add 1(=pulsar), 3(=harmonic), .5(=maybe a pulsar) to list of matches
            for vote in [1., 3., 0.5]:
                cand_pulsar = self.data[v] == vote
                for fname in self.data['fname'][cand_pulsar]:
                    self.add_candidate_to_knownpulsars(fname)
                

    def on_help(self, widget, event=None):
        """
        help
        """
        note =  "PFDviewer v0.0.1\n\n"
        note += "\tKey : 0  -- rank candidate non-pulsar\n"
        note += "\tKey : 1/p  -- rank candidate pulsar\n"
        note += "\tKey : 5/m  -- rank candidate as marginal (prob = 0.5)\n"
        note += "\tKey : h  -- rank candidate as harmonic (prob = 3.)\n"
        note += "\tKey : k  -- rank candidate as known pulsar (prob = 2.)\n"        
        note += "\tKey : b/n  -- display the previous/next candidate\n"
        note += "\tKey : c  -- cycle through possible matches\n"
        note += "\tKey : r -- reset the vote to np.nan\n"
        
        dialog = Gtk.MessageDialog(self, 0, Gtk.MessageType.INFO,
                                   Gtk.ButtonsType.OK_CANCEL, note)
        response = dialog.run()
        dialog.destroy()
        
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
            filter.set_name("text file (.txt)")
            filter.add_pattern("*.txt")
            dialog.add_filter(filter)
            filter = Gtk.FileFilter()
            filter.set_name("All *")
            filter.add_pattern("*")
            dialog.add_filter(filter)
            response = dialog.run()
            if response == Gtk.ResponseType.OK:
                self.savefile = dialog.get_filename()
#                print "File selected: " + self.savefile
            dialog.destroy()

#        print "Writing data to %s" % self.savefile
        if self.savefile == None:
            note = "No file selected. Votes not saved"
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

    def pmatch_tree_init(self):
        """
        initialize the pmatch tree(s).

        """
        for vi, v in enumerate(['name','P0 (harm)','DM','RA','DEC','vote']):
            cell = Gtk.CellRendererText()
            col = Gtk.TreeViewColumn(v, cell, text=vi)
            col.set_property("alignment", 0.5)
            if v == 'name':
                col.set_expand(True)
            else:
                col.set_expand(False)
            self.pmatch_tree.append_column(col)


    def on_pmatchwin_toggled(self, event):
        """
        show the candidate matches in their own window

        """
        if self.pmatchwin_tog.get_active():
#hide the old stuff
            self.pmatch_tree.hide()
            self.pmatch_lab.hide()
#show the new stuff
            self.pmatch_tree = self.builder.get_object('pmatch_tree1')
            if self.pmatch_tree.get_n_columns() == 0:
                self.pmatch_tree_init()
            self.pmatch_tree.connect("cursor-changed", self.on_pmatch_select_row)
            self.pmatch_tree.set_model(self.pmatch_store)
            self.pmatch_lab = self.builder.get_object('pmatch_lab1')
            self.pmatch_win.show_all()
            self.pfdtree_current()
        else:
            self.pmatch_win.hide()
            self.pmatch_tree = self.builder.get_object('pmatch_tree')
            if self.pmatch_tree.get_n_columns() == 0:
                self.pmatch_tree_init()
            self.pmatch_tree.connect("cursor-changed", self.on_pmatch_select_row)
            self.pmatch_lab = self.builder.get_object('pmatch_lab')
            self.pmatch_tree.set_model(self.pmatch_store)
            self.pmatch_tree.show_all()
            self.pmatch_lab.show_all()
            self.pfdtree_current()
        
    def on_autosave_toggled(self, event):
        """

        """
        if self.autosave.get_active():
            if self.savefile == None:
                self.on_save(event)
            else:
                self.statusbar.push(0,"Saving to %s" % self.savefile)


    def update(self, event=None):
        """
        update:
        *statusbar and make sure right cell is highlighted
        *voterbox to make sure it has all the right entries

        """        

#set cursor to first entry when loading a data file
        if event == 'load': 
            if self.data == None:
                ncand = 0
            else:
                ncand = len(self.data)

            self.pfdtree.set_cursor(0)
            if ncand == 0:
                stat = 'Please load a data file'
            else:
                stat = 'Loaded %s candidates' % ncand
            self.statusbar.push(0, stat)

    def find_matches(self):
        """
        given the selected row (pfd file), find matches to known pulsars
        and list the matches

        """
        gsel = self.pfdtree.get_selection()
        act_name = self.voters[self.active_voter]
        if gsel:
            tmpstore, tmpiter = gsel.get_selected()
        else:
            tmpiter = None

        if tmpiter != None:
            fname = '%s/%s' % (self.basedir,tmpstore.get_value(tmpiter, 1))
            store_name = tmpstore.get_value(tmpiter, 1)
# see if this path exists, update self.basedir if necessary
#            self.find_file(fname)
            try:
                pfd = pfddata(fname)
                pfd.dedisperse()
            except ValueError:
                print "prepfold can't parse %s" % fname
                pfd = None
                
            if exists(fname) and fname.endswith('.pfd') and pfd != None:

                dm = pfd.bestdm
                ra = pfd.rastr 
                dec = pfd.decstr 
                p0 = pfd.bary_p1
                if float(pfd.decstr.split(':')[0]) > 0:
                    sgn = '+'
                else:
                    sgn = '' 
                name = 'J%s%s%s' % (''.join(pfd.rastr.split(':')[:2]), sgn,\
                                        ''.join(pfd.decstr.split(':')[:2]))
#                this_pulsar = KP.pulsar(fname, name, ra, dec, p0*1e-3, dm)
                this_pulsar = KP.pulsar(fname, name, ra, dec, p0, dm)
                this_idx = self.data['fname'] == store_name
                if len(this_idx[this_idx]) > 0:
                    this_vote = self.data[act_name][this_idx][0]
                else:
                    this_vote = np.nan
                self.pmatch_tree.set_model(None)
                self.pmatch_store.clear()
                nm = 'This Candidate'
                nm = fname #basename(fname)
                self.pmatch_store.append([nm,str(np.round(this_pulsar.P0,5)),\
                                              this_pulsar.DM, this_pulsar.ra,\
                                              this_pulsar.dec, this_vote])

                sep = self.matchsep.get_value()
                matches = KP.matches(self.knownpulsars, this_pulsar, sep=sep)
                
                verbose = self.verbose_match.get_active()
                if verbose:
                    print "\n--- candidate %s (ra,dec,DM)=(%s,%s, %s) ---" %\
                        (nm, this_pulsar.ra, this_pulsar.dec, this_pulsar.DM)
                    
                for m in matches:
                    max_denom = 100
                    num, den = harm_ratio(np.round(this_pulsar.P0,5), np.round(m.P0,5), max_denom=max_denom)
                    
                    if num == 0: 
                        num = 1
                        den = max_denom
                    pdiff = abs(1. - float(den)/float(num) *this_pulsar.P0/m.P0)
                    if verbose:
                        print "position match (name, ra, dec, DM): (%s, %s, %s, %s)"\
                            % (m.name, m.ra, m.dec, m.DM)
                        
                   #don't include if this isn't a harmonic match
                    if pdiff > 1.:
                        if verbose:
                            print "  rejecting %s since pdiff = " % (m.name, pdiff)
                        continue

                    if (m.DM != np.nan) and (this_pulsar.DM != 0.):
                        #don't include pulsars with 25% difference in DM
#                        cut = 0.7*(pfd.dms.max() - pfd.dms.min())/2.
#                        if (m.DM < this_pulsar.DM - cut) or (m.DM > this_pulsar.DM + cut):
                        dDM = abs(m.DM - this_pulsar.DM)/this_pulsar.DM
                        if  dDM > .15:
                            if verbose:
                                print "  rejecting %s since Delta DM/DM = %s" % (m.name, dDM)
                            continue
                    idx = self.data['fname'] == m.name
                    if len(idx[idx]) > 0:
                        try:
                            vote = self.data[act_name][idx][0]
                        except ValueError:
                            vote = np.nan
                    else:
                        vote = np.nan
#don't add the current candidate to the list of matches
                    if basename(m.name) != basename(nm):
                        txt = "%s (%s/%s) to %4.3f" % (np.round(m.P0,5), num, den, pdiff)
                        txt += '%'
                        d = [m.name, txt, m.DM, m.ra, m.dec, vote]
                        self.pmatch_store.append(d)
                self.pmatch_tree.set_model(self.pmatch_store)
                if len(matches) > 0:
                    self.pmatch_tree.show_all()
                    self.pmatch_lab.show_all()
                    self.pmatch_tree.set_cursor(0)
                else:
                    self.pmatch_tree.hide()
#                    self.pmatch_lab.hide()
            else:
                self.pmatch_tree.hide()
#                self.pmatch_lab.hide()

    def pmatchtree_next(self):
        """
        select the next row in the "possible matches" tree
        that is a pfd file

        """
        (model, pathlist) = self.pmatch_tree.get_selection().get_selected_rows()
        fname = ''
        if len(pathlist) > 0:
            path = pathlist[0]
            tree_iter = model.get_iter(path)
            npath = model.iter_next(tree_iter)
            while npath:
                fname = model.get_value(npath, 0)
                if fname.endswith('.pfd'):
                    break
                else:
                    npath = model.iter_next(npath)
            if fname.endswith('.pfd'):
                # go back to beginning if we don't find .pfd files
                nextpath = model.get_path(npath)
                self.pmatch_tree.set_cursor(nextpath) 
                self.pmatch_tree.scroll_to_cell(nextpath, use_align=False)
            else:
                first_iter = self.pmatch_store.get_iter_first()
                if first_iter is not None:
                    path = model.get_path(first_iter)
                    self.pmatch_tree.scroll_to_cell(path)
                    self.pmatch_tree.set_cursor(path)

    def pfdtree_current(self):
        """
        trigger an event so the pfdtree gets updated

        """
        (model, pathlist) = self.pfdtree.get_selection().get_selected_rows()
#only use first selected object
        if len(pathlist) > 0:
            path = pathlist[0]
            self.pfdtree.set_cursor(path)
            self.statusbar.push(0, "")
        else:
            self.statusbar.push(0,"Please select a row")
                    
    def pfdtree_next(self, model, next_iter):
        """
        select next row in pfdtree
        """
        if next_iter:
            next_path = model.get_path(next_iter)
            self.pfdtree.set_cursor(next_path) 
            self.statusbar.push(0, "")
#        else:
#            self.statusbar.push(0,"Please select a row")
#        self.find_matches()

    def pfdtree_prev(self):
        """
        select prev row in pfdtree
        """
        (model, pathlist) = self.pfdtree.get_selection().get_selected_rows()
#only use first selected object
        if len(pathlist) > 0:
            path = pathlist[0]
            tree_iter = model.get_iter(path)
            prevn = model.iter_previous(tree_iter)
            if prevn:
                prevpath = model.get_path(prevn)
                self.pfdtree.set_cursor(prevpath)
            self.statusbar.push(0, "")
        else:
            self.statusbar.push(0,"Please select a row")
#        self.find_matches()

    def on_pfdwin_key_release_event(self, widget, event):
        key = Gdk.keyval_name(event.keyval)
        if key not in ['Control_L','Control_R','Alt_L','Alt_R']:
            self.modifier = None

    def on_update_knownpulsars(self, widget):
        """
        download the ATNF and GBNCC pulsar lists.
        If this is greater than the known_pulsars.pkl one,
        we overwrite it.

        """
        dlg = Gtk.MessageDialog(self, 0, Gtk.MessageType.INFO,
                                Gtk.ButtonsType.OK,
                                'Downloading ATNF and GBNCC pulsar databases')

        dlg.format_secondary_text('Please be patient...')
        dlg.set_modal(False)
        dlg.run()
        self.statusbar.push(0,'Downloading ATNF and GBNCC databases')
        newlist = KP.get_allpulsars()
        fout = abspath('known_pulsars.pkl')
        if self.knownpulsars == None:
            self.knownpulsars = newlist
            cPickle.dump(self.knownpulsars,open(fout,'w'))
            self.statusbar.push(0,'Saved list of %s pulsars to %s' %\
                                    (len(self.knownpulsars),fout))
        else:
            if len(newlist) > len(self.knownpulsars):
                n_new = len(newlist) - len(self.knownpulsars)
                self.knownpulsars = newlist
                self.statusbar.push(0,'Added %s new pulsars. Saving to %s' %\
                                        (n_new,fout))
                cPickle.dump(self.knownpulsars,open(fout,'w'))
            else:
                self.statusbar.push(0,'No new pulsars listed')
        dlg.destroy()
####### end MainFrame class ####

################################
## utilities

def convert(fin):
    """
    given a pfd or ps file, make the png file

    return:
    the name of the png file

    """
    global show_pfd, PFD, pyimage
    fout = ''
    if not exists(fin):
        print "Convert: can't find file %s" % abspath(fin)
        return fout

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
            #make the .ps file (later converted to .png)
            pfddir = dirname(abspath(fin))
            pfdname = basename(fin)
            full_path = abspath(fin)

            cmd = [show_pfd, '-noxwin', full_path]
            subprocess.call(cmd, shell=False,
                            stdout=open('/dev/null','w'))

            #show_pfd uses pfd.pgdev for the output filename
            #move that to filename.ps
            if (PFD != None):
                pfd = PFD(full_path)
                show_name = pfd.pgdev.replace('.ps/CPS', '') 
                for ext in ['ps', 'bestprof']:
                    pin = '%s.%s' % (show_name, ext) 
                    pout = os.path.join(pfddir, '%s.%s' %(pfdname, ext))
                    if os.path.exists(pin):
                        shutil.move(pin, pout)
                        #print "\nMoving %s to %s\n" %(pin, pout)
            else:
                #assume name was same as input filename
                #move that to the pfddir
                for ext in ['ps','bestprof']:
                #show_pfd outputs to CWD
                    fnew = abspath('%s.%s' % (pfdname, ext))
                    fold = os.path.join(pfddir, '%s.%s' % (pfdname, ext))
                    if os.path.exists(fnew):
                        shutil.move(fnew, fold)

            # assign fin to ps file so it converts to png below
            fin = abspath(os.path.join(pfddir, "%s.ps" % pfdname))
        else:
            #conversion failed
            fout = None

    if fin.endswith('.ps') and os.path.exists(fin):
        bdir = dirname(abspath(fin))
        bname = basename(fin)
        fout = os.path.join(bdir, bname.replace('.ps','.png')) 
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

        if len(data.dtype.names[1:]) == 1 and data.dtype.names[1]  == 'AI':
            #can't have data file of only 'AI'... we must be able to vote
            name = inputbox('Voter chooser',\
                                'No user voters found in %s. Add your voting name' % fname)
            data = add_voter(name, data)
    while len(data.dtype.names) == 1: #fname
        #data should have two columns.
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
            (Gtk.STOCK_OK, Gtk.ResponseType.OK    ))
    lbl = Gtk.Label(label)
    lbl.set_alignment(0, 0.5)
    lbl.show()
    dlg.vbox.pack_start(lbl, False, False, 0)
    entry = Gtk.Entry()
    entry.connect("activate", 
                      lambda ent, dlg, resp: dlg.response(resp), 
                      dlg, Gtk.ResponseType.OK)
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

def feature_predict(clf, pfd):
    """
    given a classifier and pfd file,
    return the predict_proba for the individual features and overall performance.

    Assumes 'pulsar' class in label '1'
    """
    features = ['phasebins', 'intervals', 'subbands', 'DMbins']
    avgs = {}
    for f in features:
        if clf.strategy != 'adaboost':
            avgs[f] = np.mean([c.predict_proba(pfd)[...,1][0] for c in clf.list_of_AIs \
                                   if f in c.feature])
        else:
            weights = clf.AIonAI.weights
            avgs[f] = np.sum([(c.predict_proba(pfd)[...,1][0]*2.-1)*weights[i]\
                                   for i, c in enumerate(clf.list_of_AIs) if f in c.feature])
            #note: H(x) = sign(sum_i w[i]*clf_i(x))
            #this is a small hack to get predict_proba in range 0<P<1
            max_psr = np.sum([np.abs(weights[i])\
                                  for i, c in enumerate(clf.list_of_AIs) if f in c.feature])
            avgs[f] = (avgs[f]+max_psr)/max_psr/2.
            
    avgs['overall'] = clf.predict_proba(pfd)[...,1][0]
    #note, if feature isn't present np.mean([]) == nan
    return avgs

def harm_ratio(a,b,max_denom=100):
    """
    given two numbers, find the harmonic ratio

    """
    c = fractions.Fraction(a/b).limit_denominator(max_denominator=max_denom)
    return c.numerator, c.denominator

if __name__ == '__main__':        
    
    #did we pass an input file:
    args = sys.argv[1:]
    data = None
    if len(args) > 0:
        data = args[0]#load_data(args[0])

    app = MainFrameGTK(data=data)    
    Gtk.main()
