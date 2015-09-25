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
import datetime
import fractions
import glob
import numpy as np
import os, pwd
import shutil
import subprocess
import sys
import tempfile
import ubc_AI
from os.path import abspath, basename, dirname, exists
from optparse import OptionParser

from gi.repository import Gtk, Gdk
import pylab as plt

#next taken from ubc_AI.training and ubc_AI.samples
from ubc_AI.training import pfddata
from ubc_AI.data import pfdreader 
from sklearn.decomposition import RandomizedPCA as PCA
import ubc_AI.known_pulsars as known_pulsars

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
pdmp = False
for p in os.environ.get('PATH').split(':'):
    cmd = '%s/pdmp' % p
    if exists(cmd):
        pdmp = cmd
        break
if not show_pfd:
    print "\tCouldn't find PRESTO's show_pfd executable"
    print "\t This will limit functionality"
try:
    from ubc_AI.prepfold import pfd as PFD
except(ImportError):
    print "please add PRESTO's modules to your python path for more functionality"
    PFD = None

#do not allow active voter = sort column. warn once
have_warned = False


#iter on each "n". auto-save after every 10
cand_vote = 0
#store AI_view png's in a temporary dir
tempdir = tempfile.mkdtemp(prefix='AIview_')
atexit.register(lambda: shutil.rmtree(tempdir, ignore_errors=True))
bdir = '/'.join(__file__.split('/')[:-1])
AI_path = AI_PATH = '/'.join(ubc_AI.__file__.split('/')[:-1])

class MainFrameGTK(Gtk.Window):
    """This is the Main Frame for the GTK application"""
    
    def __init__(self, data=None, tmpAI=None, spplot=False):
        Gtk.Window.__init__(self, title='pfd viewer')
        if AI_path:
            self.gladefile = "%s/pfdviewer.glade" % AI_path
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
        
        #info window stuff
        self.info_win = self.builder.get_object('info_win')
        self.info_win.set_deletable(False)
        self.info_win.connect('delete-event', lambda w, e: w.hide() or True)
        self.tmpAI_exp = self.builder.get_object('tmpAI_expander')
        self.AIview_exp = self.builder.get_object('AIview_expander')

        #palfa query window
        self.palfaqry_tog = self.builder.get_object('PALFAqry')
        self.palfaqry_win = self.builder.get_object('PALFAqry_win')
        self.palfaqry_win.set_deletable(False)
        self.palfaqry_win.connect('delete-event', lambda w, e: w.hide() or True)
        self.palfaqrybuf = self.builder.get_object('qry_view').get_buffer()
        self.palfaqry_subfile = self.builder.get_object('qry_subfile')
        self.palfa_qu = self.builder.get_object('qry_uname')
        self.palfa_qp = self.builder.get_object('qry_pwd')
        self.palfa_sampleqry = self.builder.get_object('sample_qry')
        self.palfa_sampleqry.set_active(0)
        self.qry_dwnld = self.builder.get_object('qry_dwnld')
        #use this to keep track of Query candidates
        self.qry_results = {}
        self.qry_saveloc = './'
        self.qry_savefil = '%s-qry.npy' % datetime.datetime.now().strftime('%Y_%m_%d')
        self.data_fromQry = False
#keep query in separate location, moving to self.qry_saveloc on save
        self.qrybasedir = '/dev/shm/pfdvwr.%s' % pwd.getpwuid(os.getuid())[0]
        #os.getlogin()

        #aiview window
        self.aiview_tog = self.builder.get_object('aiview')

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
        #advance to next non-voted candidate, or next in list.
        self.advance_next = self.builder.get_object('advance_next')
        self.advance_col = self.builder.get_object('advance_col')
        self.DM_limit_toggle = self.builder.get_object('DM_limit_toggle')
        self.DM_limit = self.builder.get_object('DM_limit_val')
        
        #feature-labelling stuff
        self.builder.get_object('FL_grid').hide()
        self.fl_voting_tog = self.builder.get_object('FL_voting_tog')
        self.FL_text = self.builder.get_object('FL_label')
        #when feature-labeling, do 5 votes.
        self.fl_nvote = 0

        #tmpAI window 
        self.tmpAI_win = self.builder.get_object('tmpAI_votemat')
        self.tmpAI_overall = self.builder.get_object('overall_vote')
        self.tmpAI_phasebins = self.builder.get_object('phasebins_vote')
        self.tmpAI_intervals = self.builder.get_object('intervals_vote')
        self.tmpAI_subbands = self.builder.get_object('subbands_vote')
        self.tmpAI_DMbins = self.builder.get_object('DMbins_vote')
        self.tmpAI_tog = self.builder.get_object('tmpAI_tog')
        self.tmpAI_lab = self.builder.get_object('tmpAI_lab')
        if tmpAI is not None:
            self.tmpAI = cPickle.load(open(tmpAI,'r'))
            self.tmpAI_tog.set_active(1)
            self.info_win.show_all()
            self.tmpAI_exp.set_expanded(1)
            #hack to get the checkmark "on", and preserve the tmpAI
            self.tmpAI = cPickle.load(open(tmpAI,'r'))
#            self.info_win.resize()
        else:
            self.tmpAI = None
            self.tmpAI_tog.set_active(0)
            if not self.aiview_tog.get_active():
                #then both 'info' views are off, so hide window
                self.info_win.hide()
        #keep track of seen/voted pfd's so things are quicker
        self.tmpAI_avgs= {}

        #pulsar-matching stuff
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
        #default sort on fname, later changed if number of voters > 1
        self.pfdstore.set_sort_column_id(1,1)#arg1=fname, arg2=sort/revsort

# set up the matching-pulsar tree
        self.pmatch_tree_init()

        ## data-analysis related objects
        self.voters = []
        self.savefile = None
        self.loadfile = None 
        self.knownpulsars = {}
        #ATNF, PALFA and GBNCC list of known pulsars
        if exists('%s/known_pulsars.pkl' % AI_path):
            self.knownpulsars = cPickle.load(open('%s/known_pulsars.pkl' % AI_path))
        elif exists('known_pulsars.pkl'):
            self.knownpulsars = cPickle.load(open('known_pulsars.pkl'))
        else:
            self.knownpulsars = known_pulsars.get_allpulsars()
        #if we were passed a data file, read it in
        if data != None:
            self.on_open(event='load', fin=data)
        else:
            self.data = None
        # start with default and '<new>' voters
        if self.data != None:
            self.voters = [name for name in self.data.dtype.names[1:] \
                               if not name.endswith('_FL')]
            if len(self.voters) > 1:
                self.pfdstore.set_sort_column_id(2,1)#arg1=fname, arg2=sort/revsort
                
#            self.voters = list(self.data.dtype.names[1:]) #strip off 'fname'
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
            #make active voter the first non-"AI" and non-"_FL" column
            names = [name for name in self.data.dtype.names[1:] \
                               if not name.endswith('_FL')]
            av = np.where( np.array(names != 'AI'))[0]
#            av = np.where(np.array(self.data.dtype.names[1:] != 'AI'))[0]
            if len(av) == 0:
                self.active_voter = 1
                self.statusbar.push(0, 'Warning, voting overwrites AI votes')
            else:
                name = names[av[0]]
                idx = self.voters.index(name)
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

    def on_DM_limit_toggled(self, widget):
        self.on_DM_limit_val_changed( widget)

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
            if data.ndim == 0:
                dtyp = [(name, self.data.dtype[name].str) \
                            for name in ['fname', col1, col2]]
                data = np.array([data], dtype=dtyp)
            data.sort(order=[col1,'fname'])
            limidx = data[col1] >= lim - 1e-5
            if limidx.size > 1 and np.any(limidx) and limtog:
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
            if data.ndim == 0:
                dtyp = [(name, self.data.dtype[name].str) \
                            for name in ['fname', col1]]
                data = np.array([data], dtype=dtyp)
            
            data.sort(order=[col1,'fname'])
            limidx = data[col1] >= lim - 1e-5
            if limidx.size > 1 and np.any(limidx) and limtog:
                data = data[limidx]
                self.statusbar.push(0,'Showing %s/%s candidates above %s' %
                                    (limidx.sum(),len(limidx),lim))
            else:
                self.statusbar.push(0,'No %s candidates > %s. Showing all' % (col1, lim))
            for vi, v in enumerate(data[::-1]):
                v0, v1 = v
                self.pfdstore.append((vi,v0,float(v1),float(v1)))
                    
        self.pfdtree.set_model(self.pfdstore)
        self.find_matches()


    def on_DM_limit_val_changed(self, widget):
        """
        set limit on the DM of candidadtes to display
        """
        limtog = self.DM_limit_toggle.get_active()
        try:
            lim = float(self.DM_limit.get_text())
        except:
            return

        if not limtog: 
            self.on_view_limit_changed(widget)
            return

        col1 = self.col1.get_active_text()
        col2 = self.col2.get_active_text()
        idx1 = self.col_options.index(col1)
        if col2 == None:
            col2 = col1
            self.col2.set_active(idx1)
        idx2 = self.col_options.index(col2)

        if len(self.pfdstore) == 0:return None #do nothing

        #if idx1 != idx2:
            #dtyp = [('n', '<i8')] + [(name, self.data.dtype[name].str) for name in ['fname', col1, col2]]
        #else:
            #dtyp = [('n', '<i8')] + [(name, self.data.dtype[name].str) for name in ['fname', col1]]

        removecount = 0
        for i, one in enumerate(self.pfdstore):
            onesdm = pfddata(one[1]).bestdm
            if not onesdm > lim:
                self.pfdstore.remove(one.iter)
                removecount += 1 
        print "removed %s candidates with DM < %s" % (removecount, lim)
        self.statusbar.push(0,'removed %s candidates with DM < %s.' % (removecount, lim))

        
        return 
#turn off the model first for speed-up
        """
        #if idx1 != idx2:
            #data = self.data[['fname',col1,col2]]
            #if data.ndim == 0:
                #dtyp = [(name, self.data.dtype[name].str) \
                            #for name in ['fname', col1, col2]]
                #data = np.array([data], dtype=dtyp)
            #data.sort(order=[col1,'fname'])
            #limidx = data[col1] >= lim - 1e-5
            #if limidx.size > 1 and np.any(limidx) and limtog:
                #data = data[limidx]
                #self.statusbar.push(0,'Showing %s/%s candidates above %s in DM' %
                                    #(limidx.sum(),len(limidx),lim))
            #else:
                #self.statusbar.push(0,'No %s candidates > %s in DM. Showing all' % (col1, lim))
            #for vi, v in enumerate(data[::-1]):
                #d = (vi,) + v.tolist() 
                #self.pfdstore.append(d)
        #else:
            #data = self.data[['fname',col1]]
            #if data.ndim == 0:
                #dtyp = [(name, self.data.dtype[name].str) \
                            #for name in ['fname', col1]]
                #data = np.array([data], dtype=dtyp)
            
            data.sort(order=[col1,'fname'])
            limidx = data[col1] >= lim - 1e-5
            if limidx.size > 1 and np.any(limidx) and limtog:
                data = data[limidx]
                self.statusbar.push(0,'Showing %s/%s candidates above %s in DM' %
                                    (limidx.sum(),len(limidx),lim))
            else:
                self.statusbar.push(0,'No %s candidates > %s in DM. Showing all' % (col1, lim))
            for vi, v in enumerate(data[::-1]):
                v0, v1 = v
                self.pfdstore.append((vi,v0,float(v1),float(v1)))
                    
        self.pfdtree.set_model(self.pfdstore)
        self.find_matches()
        """

    def on_pfdwin_key_press_event(self, widget, event):
        """
        controls keypresses on over-all window

        #recently added "feature-label" voting, where
        #we cycle through the subplots before advancing

        """
        global cand_vote, have_warned

        #are we doing feature-label voting?
        FL = False
        if self.fl_voting_tog.get_active():
            FL = True

        #key codes which change the voting data
        votes = {'1':1., 'p':1.,    #pulsar
                 'r': np.nan,      #reset to np.nan
                 '5':.5, 'm':.5,   #50/50 pulsar/rfi
                 'k':2.,           #known pulsar
                 'h':3.,           #harmonic of known
                 '0':0.            #rfi (not a pulsar)
                 }
        #checkboxes for FL voting (vote: glade checkbox)
        FL_votes = {0: 'FL_overall',
                    1: 'FL_profile',
                    2: 'FL_intervals',
                    3: 'FL_subbands',
                    4: 'FL_DMcurve'
                    }
                 
        key = Gdk.keyval_name(event.keyval)
        ctrl = event.state &\
            Gdk.ModifierType.CONTROL_MASK
        if self.active_voter:
            act_name = self.voters[self.active_voter]
            if self.fl_voting_tog.get_active():
                act_name += '_FL'
        else:
            act_name = 'AI'
            
#don't allow voting/actions if sort_column = active voter
        col1 = self.col1.get_active_text()
        col2 = self.col2.get_active_text()
        sort_id = self.pfdstore.get_sort_column_id()[0]
        #0=number, 1=fname, 2=col1, 3=col2
        if (sort_id == 2) and (act_name == col1) and key in votes:
            note = "Note. Voting disabled when active voter = sort column. \n"
            note += "Try sorting by filename"
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

        #if we've made it this far than we can record the vote...
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
            #look for next non-ranked candidates
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
            self.pfdtree_next(model, next_iter, FL=FL)
        elif key == 'b':
            self.pfdtree_prev(FL=FL)
        elif key == 'a':
            #toggle aiview
            d = self.aiview_tog.get_active()
            self.aiview_tog.set_active(not(d))
        elif key == 'd':
            #download the candidate if we are in QRY mode
            if self.data_fromQry:
                #get the filename
                fname = self.pfdstore[this_iter][1]
                if not os.path.exists(os.path.join(self.qrybasedir,fname)):
                    self.PALFA_download_qry(fname)
        elif key == 'Delete':
            # remove this file from the list of tracked files
            if this_iter is not None:
                fname = self.pfdstore[this_iter][1]
                next_path = model.get_path(next_iter)
                self.remove_fname(fname) 
                self.pfdtree.set_cursor(next_path)
        elif key in ['Left', 'Right']:
            # FL and 'left' goes back a FL 
            if key == 'Left':
                self.fl_nvote = max(0, self.fl_nvote - 1)
            elif key == 'Right':
                self.fl_nvote = min(4, self.fl_nvote + 1)
            self.FL_color()
            a = FL_votes[self.fl_nvote].strip('FL_')
            self.FL_text.set_text('FL vote:\n (%s)' % a)

        #data-related (needs to be loaded)
        if self.data != None:
            if key in votes:
                cand_vote += 1

                #download the candidate if we are in QRY mode
                if self.data_fromQry:
                    #get the filename
                    fname = self.pfdstore[this_iter][1]
                    if not os.path.exists(os.path.join(self.qrybasedir,fname)):
                        self.PALFA_download_qry(fname)
                value = votes[key]

                #FL only votes 0/1
                if FL:
                    value = int(value)
                    if value in [0,1]:
                        #set the checkbox
                        o = self.builder.get_object(FL_votes[self.fl_nvote])
                        o.set_active(value)
                        fname = self.pfdstore_set_value(value, this_iter=this_iter, \
                                                            return_fname=True, FL=FL)
                else:
                    fname = self.pfdstore_set_value(value, this_iter=this_iter, \
                                                        return_fname=True)
                
                if key in ['1', 'p', 'm', '5', 'h', 'k']:
                    if FL and self.fl_nvote == 0:
                        self.add_candidate_to_knownpulsars(fname)
                    elif not(FL):
                        self.add_candidate_to_knownpulsars(fname)
                if FL and value in [0,1]:
                    self.fl_nvote = (self.fl_nvote + 1) % 5
                    self.FL_color()
                    a = FL_votes[self.fl_nvote].strip('FL_')
                    self.FL_text.set_text('FL vote:\n (%s)' % a)


                #advance to the next candidate?
                if next_iter is not None:
                    if FL and value in [0, 1]:
                        if (self.fl_nvote % 5 == 0):
                        #then we've done all the FL voting
                            self.pfdtree_next(model, next_iter, FL=FL)
                    elif (not FL):
                        self.pfdtree_next(model, next_iter, FL=FL)

            elif key == 'c':
                # cycle between ranked candidates
                self.pmatchtree_next()

            if cand_vote//10 == 1:
                if self.autosave.get_active():
                    self.on_save()
                else:
                    self.statusbar.push(0,'Remember to save your output')

            awards = [30, 75, 150, 250, 400, 600, 1000]
            if 0 and (cand_vote in awards) | ( (cand_vote > max(awards)) & (cand_vote%500 == 0) ):
                try:
                    idx = awards.index(cand_vote) 
                    level = idx + 1
                except(ValueError):
                    level = cand_vote//500 + len(awards) - max(awards)//500
                note = 'Congratulations  %s. You are a level %s classifier!\n' % (act_name, level)
                
                if level < len(awards):
                    nl = awards[level] - awards[idx]
                else:
                    nl = 500
                note += 'Only %s votes to the next level, and possibly a free coffee from Aaron and Weiwei.\n' % nl
                note += 'Save a screenshot of this window for proof.' 
                    
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
#            this_pulsar = known_pulsars.pulsar(fname, name, ra, dec, p0*1e-3, dm)
            this_pulsar = known_pulsars.pulsar(fname, name, ra, dec, p0, dm, catalog='local')
                
            self.knownpulsars[fname] = this_pulsar


    def dataload_update(self):
        """
        update the pfdstore whenever we load in a new data file
        
        set columns to "fname, AI, [1st non-AI voter... if exists]"

        """
        if self.active_voter:
            act_name = self.voters[self.active_voter]
#update the Treeview... 
            if self.data is not None:

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
                    if data.ndim == 0:
                        dtyp = [(name, self.data.dtype[name].str) \
                                    for name in ['fname', col1, col2]]
                        data = np.array([data], dtype=dtyp)
                    data.sort(order=[col1,'fname'])
                    limidx = data[col1] >= lim - 1e-5
                    if limidx.size > 1 and np.any(limidx) and limtog:
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
                    if data.ndim == 0:
                        dtyp = [(name, self.data.dtype[name].str) \
                                    for name in ['fname', col1]]
                        data = np.array([data], dtype=dtyp)
                    data.sort(order=[col1,'fname'])
                    limidx = data[col1] >= lim - 1e-5
                    if limidx.size > 1 and np.any(limidx) and limtog:
                        data = data[limidx]
                        self.statusbar.push(0,'Showing %s/%s candidates above %s' %
                                            (limidx.sum(),len(limidx),lim))
                    else:
                        self.statusbar.push(0,'No %s candidates > %s. Showing all.' % (col1, lim))

                    for vi, v in enumerate(data[::-1]):
                            v0, v1 = v
                            self.pfdstore.append((vi,v0,float(v1),float(v1)))
                        
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
                    if data.ndim == 0:
                        dtyp = [(name, self.data.dtype[name].str) \
                                    for name in ['fname', col1, col2]]
                        data = np.array([data], dtype=dtyp)
                    data.sort(order=[col1,'fname'])
                    limidx = data[col1] >= lim - 1e-5
                    if limidx.size > 1 and np.any(limidx) and limtog:
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
                    if data.ndim == 0:
                        dtyp = [(name, self.data.dtype[name].str) \
                                    for name in ['fname', col1]]
                        data = np.array([data],dtype=dtyp)
                    data.sort(order=[col1,'fname'])
                    limidx = data[col1] >= lim - 1e-5
                    if limidx.size > 1 and np.any(limidx) and limtog:
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
        
        if feature-label voting, load up the votes

        """
        gsel = self.pfdtree.get_selection()
        if gsel:
            tmpstore, tmpiter = gsel.get_selected()
        else:
            tmpiter = None
        FL = False
        if self.fl_voting_tog.get_active():
            FL = True

        ncol = self.pfdstore.get_n_columns()
        if tmpiter != None:
            name = tmpstore.get_value(tmpiter,1)
            if self.data_fromQry:
                idx = np.where(self.qry_results['filename'] == basename(name))[0]
                if len(idx) == 1 and self.qry_results['keep'][idx]:
                    basedir = self.qry_saveloc
                else:
                    basedir = self.qrybasedir
            else:
                basedir = self.basedir
            fname = os.path.join(basedir, name)
            
#are we displaying the prediction from a tmpAI?            
            if exists(fname) and fname.endswith('.pfd') \
                    and (self.tmpAI != None) and self.tmpAI_tog.get_active():
                if fname not in self.tmpAI_avgs:
                    pfd = pfdreader(fname)
                    #pfd = pfddata(fname)
                    #pfd.dedisperse()
                    avgs = feature_predict(self.tmpAI, pfd)
                    self.tmpAI_avgs[fname] = avgs
                else:
                    avgs = self.tmpAI_avgs[fname]
                self.update_tmpAI_votemat(avgs)
                disp_apnd = '(tmpAI: %0.3f)' % (avgs['overall'])
            elif (self.tmpAI != None) and self.tmpAI_tog.get_active():
                avgs = {'phasebins':np.nan,'subbands':np.nan,'intervals':np.nan,\
                            'DMbins':np.nan,'overall':np.nan}
                self.update_tmpAI_votemat(avgs)
                disp_apnd = '(pfd not found)'
            else:
                disp_apnd = ''

# find/create png file from input file
            #try:
            fpng = self.create_png(fname)
            #except:
                #print 'failed to create png file for %' % (fname)

            #update the basedir if necessary 
            if not exists(fpng):
                fname = self.find_file(fname)
                fpng = self.create_png(fname)
            
            #we are not doing "AI view" of data
            if not self.aiview_tog.get_active():
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

            #load up the feature-label votes
            if FL:
                self.fl_nvote = 0 
                idx = np.where(self.data['fname'] == name)
                act_name = self.voters[self.active_voter] + '_FL'
                kv = self.data[act_name][idx][0]
                for i, v in enumerate(['FL_overall', 'FL_profile', 'FL_intervals',\
                                           'FL_subbands','FL_DMcurve']):
                    o = self.builder.get_object(v)
                    o.set_active(kv[i])
                self.FL_color()
            self.find_matches()

    def create_png(self, fname):
        """
        given some pfd or ps file, create the png file
        for i, t in enumerate(self.pfdstore):
            print i, list(t)
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
        elif fname.endswith('.ar2') or fname.endswith('.ar'):
            fpng = '%s.png' % fname
            if not exists(fpng):
                fpng = convert(fname)
        elif fname.endswith('.spd'):
            fpng = '%s.png' % fname
            if not exists(fpng):
                fpng = convert(fname)
                #if fps.endswith('.ps'):
                    #fpng = convert(fps)
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
                if self.data_fromQry:
                    idx = np.where(self.qry_results['filename'] == basename(fname))[0]
                    if len(idx) == 1 and self.qry_results['keep'][idx]:
                        self.qry_saveloc = dialog.get_filename()
                        basedir = self.qry_saveloc
                    else:
                        self.qrybasedir = dialog.get_filename()
                        basedir = self.qrybasedir
                else:
                    self.basedir = dialog.get_filename()
                    basedir = self.basedir 
            else:
                if self.data_fromQry:
                    idx = np.where(self.qry_results['filename'] == basename(fname))[0]
                    if len(idx) == 1 and self.qry_results['keep'][idx]:
                        self.qry_saveloc = dialog.get_filename()
                        basedir = self.qry_saveloc
                    else:
                        self.qrybasedir = dialog.get_filename()
                        basedir = self.qrybasedir
                else:
                    self.basedir = './'
                    basedir = self.basedir 
            dialog.destroy()
            fname = basename(fname)
            
            fname = os.path.join(abspath(basedir), fname)
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

        Notes:
        * We save the files in /tmp/AIview*,
        * If tmpAI is not None, we use its' pca/nbins
          (we take the first of each type of subplot in the list_of_AIs)

        """
        
        #pfd = pfddata(fname)
        pfd = pfdreader(fname)
        plt.figure(figsize=(8,5.9))
        vals = [('pprof_nbins', 'pprof_pcacomp','phasebins'), #pulse profile
                ('si_nbins', 'si_pcacomp', 'subbands'),       #frequency subintervals
                ('pi_bins', 'pi_pcacomp', 'intervals'),        #pulse intervals
                ('dm_bins', 'dm_pcacomp', 'DMbins')         #DM-vs-chi2
                ]
        AIview = []
        for subplt, inp in enumerate(vals):
            ai = None
            pca = None
            nbins = None
            npca_comp = None
            #use the tmpAI values if tmpAI is loaded
            if self.tmpAI is not None:
                for tai in self.tmpAI.list_of_AIs:
                    if inp[2] in tai.feature:
                        ai = tai
                        nbins = tai.feature[inp[2]]
                        if ai.use_pca: 
                            pca = ai.pca
                            npca_comp = ai.pca.n_components
            if nbins is not None:
                self.builder.get_object(inp[0]).set_text(str(nbins))
                self.builder.get_object(inp[0]).set_editable(0)
            else:
                self.builder.get_object(inp[0]).set_editable(1)
                nbins = self.builder.get_object(inp[0]).get_text()
            if npca_comp is not None:
                self.builder.get_object(inp[1]).set_text(str(npca_comp))
                self.builder.get_object(inp[1]).set_editable(0)
            else:
                self.builder.get_object(inp[1]).set_editable(1)
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
                if npca_comp and (pca is None):
                    pca = PCA(n_components=npca_comp)
                    pca.fit(data)
                    pcadata = pca.transform(data)
                    data = pca.inverse_transform(pcadata)
                elif pca is not None:
                    pcadata = pca.transform(data)
                    data = pca.inverse_transform(pcadata)
                ax.plot(data)
                ax.set_xlim([0,len(data)])
                ax.set_title('pulse profile (bins, pca) = (%s,%s)'%(nbins,npca_comp))
            elif subplt == 1:
                ax = plt.subplot2grid((3,2), (0,1), rowspan=2)
                data = pfd.getdata(subbands=nbins)
                if npca_comp and (pca is None):
                    #note: PCA is best set when fed many samples, not one
                    pca = PCA(n_components=npca_comp)
                    rd = data.reshape(nbins,nbins)
                    pca.fit(rd)
                    data = pca.inverse_transform(pca.transform(rd)).flatten()
                elif pca is not None:
#                    rd = data.reshape(nbins,nbins)
                    data = pca.inverse_transform(pca.transform(data)).flatten()
                ax.imshow(data.reshape(nbins, nbins),
                          cmap=plt.cm.gray_r)
                ax.set_title('subbands (bins, pca) = (%s,%s)'%(nbins,npca_comp))
            elif subplt == 2:
                ax = plt.subplot2grid((3,2), (1,0), rowspan=2)
                data = pfd.getdata(intervals=nbins)
                if npca_comp and (pca is None):
                    #note: PCA is best set when fed many samples, not one
                    pca = PCA(n_components=npca_comp)
                    rd = data.reshape(nbins,nbins)
                    pca.fit(rd)
                    data = pca.inverse_transform(pca.transform(rd)).flatten()
                elif pca is not None:
#                    rd = data.reshape(nbins,nbins)
                    data = pca.inverse_transform(pca.transform(data)).flatten()
                ax.imshow(data.reshape(nbins,nbins),
                                cmap=plt.cm.gray_r)
                ax.set_title('intervals (bins, pca) = (%s,%s)'%(nbins,npca_comp))
            elif subplt == 3:
                ax = plt.subplot2grid((3,2), (2,1))
                data = pfd.getdata(DMbins=nbins)
                if npca_comp and (pca is None):
                    pca = PCA(n_components=npca_comp).fit(data)
                    pcadata = pca.transform(data)
                    data = pca.inverse_transform(pcadata)
                elif pca is not None:
                    pcadata = pca.transform(data)
                    data = pca.inverse_transform(pcadata)
                ax.plot(data)
                ax.set_title('DM curve (bins, pca) = (%s,%s)'%(nbins,npca_comp))
            ax.set_yticklabels([])
            if subplt != 3:
                majticks = np.linspace(0,nbins,5).astype(int)
                majlab = [0, .25, .5, .75, 1]
                ax.set_xticks(majticks)
                ax.set_xticklabels(majlab)
            else:
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

        
    def pfdstore_set_value(self, value, this_iter=None, return_fname=False, FL=False):
        """
        update the pfdstore value for the given path
        
        Args:
        Value: the voter prob (ranking) to assign
        return_fname: return the filename of the pfd file
        FL : are we doing feature labeling? then use self.fl_nvote

        """
        if this_iter != None:
#update self.data (since dealing with TreeStore blows my mind)
            n, fname, x, oval = self.pfdstore[this_iter]
            col1 = self.col1.get_active_text()
            col2 = self.col2.get_active_text()

            idx = self.data['fname'] == fname
            if self.active_voter:
                act_name = self.voters[self.active_voter]
                if FL:
                    act_name += '_FL'
                    value = int(value)
                    kv = self.data[act_name][idx][0]
                    kv[self.fl_nvote] = value
                    self.data[act_name][idx] = kv
                else:
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
                    if self.data_fromQry:
                        basedir = self.qrybasedir
                    else:
                        basedir = self.basedir
                    fname = os.path.join(basedir, fname)
                    if not self.aiview_tog.get_active():
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
                    if exists(fname) and fname.endswith('.pfd') and (self.tmpAI is not None) and self.tmpAI_tog.get_active():
                        if fname not in self.tmpAI_avgs:
                            pfd = pfdreader(fname)
                            #pfd = pfddata(fname)
                            #pfd.dedisperse()
                            avgs = feature_predict(self.tmpAI, pfd)
                            self.tmpAI_avgs[fname] = avgs
                        else:
                            avgs = self.tmpAI_avgs[fname]
                        self.update_tmpAI_votemat(avgs)
                        disp_apnd = '(tmpAI: %0.3f)' % avgs['overall']
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
        if self.tmpAI_tog.get_active() and (self.tmpAI is None):
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
            self.tmpAI_exp.set_expanded(0)
#            self.info_win.resize()
            if not self.AIview_exp.get_expanded():
                #then both 'info' view are off, so hide window
                self.info_win.hide()
        else:
            self.info_win.show_all()
            self.tmpAI_exp.set_expanded(1)
#            self.info_win.resize()

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

    def on_FL_voting_toggled(self, event):
        FL_votes = {0: 'FL_overall',
                    1: 'FL_profile',
                    2: 'FL_intervals',
                    3: 'FL_subbands',
                    4: 'FL_DMcurve'
                    }
        #if we're turning this on, make sure the current voter has a '_FL' column
        if self.active_voter:
            act_name = self.voters[self.active_voter]
            fl_actname = '%s_FL' % act_name
            if fl_actname not in self.data.dtype.names:
                self.data = add_voter(fl_actname, self.data, this_dtype='5i8')
        o = self.builder.get_object('FL_grid')

        if self.fl_voting_tog.get_active():
            o.show_all()
            #set the checkmarks:
            (model, pathlist) = self.pfdtree.get_selection().get_selected_rows()
#only use first selected object
            if len(pathlist) != 0:
                path = pathlist[0]
                this_iter = model.get_iter(path)
                fname = self.pfdstore[this_iter][1]
                idx = self.data['fname'] == fname
                act_name = self.voters[self.active_voter] + '_FL'
                kv = self.data[act_name][idx][0]
                self.fl_nvote = 0
                for i, v in enumerate(['FL_overall', 'FL_profile', 'FL_intervals',\
                                           'FL_subbands','FL_DMcurve']):
                    o = self.builder.get_object(v)
                    o.set_active(kv[i])
                self.FL_color()
            self.FL_text.set_text('FL vote:\n (Overall)')
        else:
            o.hide()
        pass

    def FL_color(self):
        """
        set the color of the active FL vote
        
        Note: we turn this off for now b/c the behaviour isn't good
        """
        if False:
          for i, v in enumerate(['FL_overall', 'FL_profile', 'FL_intervals',\
                                           'FL_subbands','FL_DMcurve']):
            o = self.builder.get_object(v)
            if i == self.fl_nvote:
                o.modify_fg(Gtk.StateType.NORMAL, Gdk.color_parse('red'))
            else:
                o.modify_fg(Gtk.StateType.NORMAL, Gdk.color_parse('black'))



    def on_aiview_toggled(self, event):
        """
        display or destroy the AI_view parameters window

        """
        if self.aiview_tog.get_active():
            self.info_win.show_all()
            self.AIview_exp.set_expanded(1)
#            self.info_win.resize()
        else:
            if not self.tmpAI_tog.get_active():
                #then both 'info' views are off, hide window
                self.info_win.hide()
            self.AIview_exp.set_expanded(0)
#            self.info_win.resize()
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
        
        add '_FL' if necessary

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

        if self.active_voter:
            act_name = self.voters[self.active_voter]
            if self.fl_voting_tog.get_active():
                act_name += '_FL'
            if self.data is not None:
                if act_name not in self.data.dtype.names:
                    self.data = add_voter(act_name, self.data, this_dtype='5i8')
            #print self.data[act_name]


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
            #cleanup files in RAM
            if os.path.exists('/dev/shm/pfdvwr'):
                shutil.rmtree('/dev/shm/pfdvwr',ignore_errors=True)
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
            elif response == Gtk.ResponseType.CANCEL:
                print "Cancel clicked"
                fname = None
            dialog.destroy()
        else:
            fname = fin
                
        if fname:
            self.data = load_data(fname)
            if self.data is None:
                print "Failed to open file"
                return

            self.loadfile = fname
            oldvoters = self.voters
            self.voters = [name for name in self.data.dtype.names[1:]\
                               if not name.endswith('_FL')]
#            self.voters = list(self.data.dtype.names[1:]) #0=fnames
            for v in self.voters:
                if v not in self.col_options:
                    self.col_options.append(v)
                    self.col1.append_text(v)
                    self.col2.append_text(v)

#<new> is a special case used to add new voters
            if '<new>' not in self.voters:
                self.voters.insert(0,'<new>')

            self.active_voter = 1

            #add new voters to the voterbox and datacolumn
            for v in self.voters:
                self.voterbox.append_text(v)
                if v not in oldvoters:
                    if v != '<new>':
                        self.data = add_voter(v, self.data)

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
            self.statusbar.push(0,'Loaded %s candidates' % self.data.size)
        self.dataload_update()
        self.pfdtree.set_cursor(0)

        if self.knownpulsars == None:
            self.statusbar.push(0,'Downloading ATNF, PALFA and GBNCC list of known pulsars')
            self.knownpulsars = known_pulsars.get_allpulsars()
            self.statusbar.push(0,'Downloaded %s known pulsars for x-ref'\
                                % len(self.knownpulsars))

#add all the candidates ranked as pulsars to the list of known_pulsars
        if self.data is not None:
            for v in self.data.dtype.names[1:]:
            #skip all feature-label voters
                if v.endswith('_FL'): continue 
            #add 1(=pulsar), 2(=known), 3(=harmonic), .5(=maybe a pulsar) to list of matches
                for vote in [1., 2., 3., 0.5]:
                    cand_pulsar = self.data[v] == vote
                    if self.data.size == 1:
                        fname = self.data['fname']
                        self.add_candidate_to_knownpulsars(str(fname))
                    else:
                        fnames = self.data['fname'][cand_pulsar]
                        for fname in fnames:#self.data['fname'][cand_pulsar]:
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
        note += "\tKey : a -- toggle AIview\n"
        note += "\tKey : d -- download candidate from PALFA database (after query)\n"
        note += "\tKey : Delete -- remove candidate from file/list\n"
        note += "\tKey : Left -- Go to previous sub-vote in FL voting\n"
        note += "\tKey : Right -- Go to next sub-vote in FL voting"
        
        dialog = Gtk.MessageDialog(self, 0, Gtk.MessageType.INFO,
                                   Gtk.ButtonsType.OK, note)
        response = dialog.run()
        dialog.destroy()
        
    def on_save(self, event=None):
        """
        save the data.

        """
        if self.savefile == None:
            dialog = Gtk.FileChooserDialog("choose an output file.", self,
                                           Gtk.FileChooserAction.SAVE,
                                           (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                                            Gtk.STOCK_OPEN, Gtk.ResponseType.OK))
            if self.loadfile:
                suggest = os.path.splitext(self.loadfile)[0] + '.txt'
            else:
                suggest = 'voter_rankings.txt'
            dialog.set_current_name(suggest)
            filter = Gtk.FileFilter()
            filter.set_name("text file (.txt)")
            filter.add_pattern("*.txt")
            dialog.add_filter(filter)
            filter = Gtk.FileFilter()
            filter.set_name("numpy file (*.npy)")
            filter.add_pattern("*.npy")
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
        elif self.savefile.endswith('.npy') or self.savefile.endswith('.pkl'):
            note = 'Saved to numpy file %s' % self.savefile
            if self.data_fromQry:
                np.save(self.savefile, self.data[self.qry_results['keep']])
            else:
                np.save(self.savefile, self.data)
            if not self.autosave.get_active():
                print note
            self.statusbar.push(0,note)
        else:
            note = 'Saved to textfile %s' % self.savefile
            fout = open(self.savefile,'w')
            l1 = '#'
            names = [name for name in self.data.dtype.names if not name in ['featurevoter_FL', 'featurevoter' ]]
            if 'featurevoter_FL' in self.data.dtype.names:
                self.data['Overall'] = self.data['featurevoter_FL'][...,0]
                self.data['Profile'] = self.data['featurevoter_FL'][...,1]
                self.data['Interval'] = self.data['featurevoter_FL'][...,2]
                self.data['Subband'] = self.data['featurevoter_FL'][...,3]
                self.data['DMCurve'] = self.data['featurevoter_FL'][...,4]
            outputdata = self.data[names]
            l1 += ' '.join(names)
            l1 += '\n'
            fout.writelines(l1)
            for n, row in enumerate(outputdata):
                if self.data_fromQry:
                    if self.qry_results['keep'][n]:
                        for ri, rv in enumerate(row):
                            if ri > 0:
                                fout.write("%s " % ",".join(rv.astype('str').tolist()))
                            else:
                                fout.write("%s " % rv)
                        fout.write("\n")
                else:
                    for ri, rv in enumerate(row):
                        if ri > 0:
                            fout.write("%s " % ",".join(rv.astype('str').tolist()))
                        else:
                            fout.write("%s " % rv)
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
            if self.data_fromQry:
                basedir = self.qrybasedir
            else:
                basedir = self.basedir
            fname = '%s/%s' % (basedir,tmpstore.get_value(tmpiter, 1))
            store_name = tmpstore.get_value(tmpiter, 1)
            if exists(store_name):
                fname = store_name
            else:
                fname = '%s/%s' % (basedir,tmpstore.get_value(tmpiter, 1))
                if not exists(fname):
                    fname = self.find_file(fname)
                    if not exists(fname):
                        print "can't find file %s" % fname
                        print "known pulsar match won't work for this candidate."
# see if this path exists, update self.basedir if necessary
            pfd = None
            if fname.endswith('.pfd'):
                try:
                    pfd = pfddata(fname)
                    pfd.dedisperse()
                    dm = pfd.bestdm
                    ra = pfd.rastr 
                    dec = pfd.decstr 
                    p0 = pfd.bary_p1
                except(IOError, ValueError):pass
            arch = None
            if fname.endswith('.ar2') or fname.endswith('.ar'):
                try:
                    import psrchive
                    arch = psrchive.Archive_load(fname)
                    arch.dedisperse()
                    coord = arch.get_coordinates()
                    dm = arch.get_dispersion_measure()
                    ra, dec = coord.getHMSDMS().split(' ')
                    p0 = arch[0].get_folding_period()
                except(IOError, ValueError):pass
            spd = None
            if fname.endswith('.spd'):
                try:
                    from ubc_AI.singlepulse import SPdata
                    spd = SPdata(fname)
                    dm = spd.dm
                    ra = spd.ra
                    dec = spd.dec
                    #p0 = spd.period
                    p0 = 0.
                except(IOError, ValueError):pass
            
            if exists(fname) and (pfd != None or arch != None or spd != None):
                if float(dec.split(':')[0]) > 0:
                    sgn = '+'
                else:
                    sgn = '' 
                name = 'J%s%s%s' % (''.join(ra.split(':')[:2]), sgn,\
                                        ''.join(dec.split(':')[:2]))
#                this_pulsar = known_pulsars.pulsar(fname, name, ra, dec, p0*1e-3, dm)
                this_pulsar = known_pulsars.pulsar(fname, name, ra, dec, p0, dm)
                this_idx = np.array(self.data['fname'] == store_name)
                if this_idx.size == 1:
                    this_idx = np.array([this_idx])
                if len(this_idx[this_idx]) > 0:
                    data = self.data[act_name]
                    if this_idx.size == 1:
                        data = np.array([data])
                    this_vote = data[this_idx][0]
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
                matches = known_pulsars.matches(self.knownpulsars, this_pulsar, sep=sep)
                
                verbose = self.verbose_match.get_active()
                if verbose:
                    print "\n--- candidate %s (ra,dec,DM)=(%s,%s, %s) ---" %\
                        (nm, this_pulsar.ra, this_pulsar.dec, this_pulsar.DM)
                    
                #we do this loop first, adding non-local candidates, then local ones
                match_nonlocal_loc = [v for v in matches if v.catalog != 'local']
                for v in matches:
                    if v.catalog == 'local':
                        match_nonlocal_loc.append(v)
                for m in match_nonlocal_loc:
                    max_denom = 100
                    num, den = harm_ratio(np.round(this_pulsar.P0,5), np.round(m.P0,5), max_denom=max_denom)
                    if num == 0 and den == 0:return
                    
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
                    idx = np.array(self.data['fname'] == m.name)
                    if idx.size == 1:
                        idx = np.array([idx])
                    if len(idx[idx]) > 0:
                        try:
                            data = self.data[act_name]
                            if idx.size == 1:
                                data = np.array([data])
                            vote = data[idx][0]
                        except ValueError:
                            vote = np.nan
                    else:
                        vote = np.nan
#don't add the current candidate to the list of matches
                    if basename(m.name) != basename(nm):
                        txt = "%s (%s/%s) to %4.3f" % (np.round(m.P0,5), num, den, pdiff*100.)
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
                    
    def pfdtree_next(self, model, next_iter, FL=False):
        """
        select next row in pfdtree
        
        Optional:
        FL : if feature-labelling, set the checkboxes to current vote
            otherwise 0
        """
        if next_iter:
            next_path = model.get_path(next_iter)
            self.pfdtree.set_cursor(next_path) 
            self.statusbar.push(0, "")
            #reset the number of FL votes, and the checkboxes to voter-defined state
            if FL:
                FL_votes = ['FL_overall', 'FL_profile', 'FL_intervals',\
                                'FL_subbands','FL_DMcurve']
                self.fl_nvote = 0
                fname = model.get_value(next_iter, 1)
                idx = np.where(self.data['fname'] == fname)
                act_name = self.voters[self.active_voter] + '_FL'
                kv = self.data[act_name][idx][0]
                for i, v in enumerate(FL_votes):
                    o = self.builder.get_object(v)
                    o.set_active(kv[i])
                a = FL_votes[self.fl_nvote].strip('FL_')
                self.FL_text.set_text('FL vote:\n (%s)' % a)
                self.FL_color()
        else:
            self.statusbar.push(0,"Please select a row")

    def pfdtree_prev(self, FL=False):
        """
        select prev row in pfdtree
        Optional:
        FL : if doing feature-labelling, we see if we've voted on this candidate
             and set the checkboxes appropriately
        
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

                if FL:
                    FL_votes = ['FL_overall', 'FL_profile', 'FL_intervals',\
                                    'FL_subbands','FL_DMcurve']
                    self.fl_nvote = 0
                    fname = model.get_value(prevn, 1)
                    idx = np.where(self.data['fname'] == fname)
                    act_name = self.voters[self.active_voter] + '_FL'
                    kv = self.data[act_name][idx][0]
                    for i, v in enumerate(FL_votes):
                        o = self.builder.get_object(v)
                        o.set_active(kv[i])
                    a = FL_votes[self.fl_nvote].strip('FL_')
                    self.FL_text.set_text('FL vote:\n (%s)' % a)
                    self.FL_color()
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
        newlist = known_pulsars.get_allpulsars()
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

    def on_sampleqry_changed(self, event):
        """
        load a template PALFA query
        """
        tmplts = {'by candidate id': "SELECT * FROM PDM_Candidate_Binaries_Filesystem as t where t.pdm_cand_id = %s",
                  'recent rfi': "SELECT DISTINCT t1.pdm_cand_id from pdm_candidates as t1 with(nolock)  RIGHT JOIN headers as t4 with(nolock) on t1.header_id=t4.header_id AND t4.obsType in ('Mock','wapp')  LEFT OUTER JOIN pdm_classifications as t2 with(nolock) ON t2.pdm_cand_id=t1.pdm_cand_id AND t2.person_id=30 AND t2.pdm_class_type_id=1  LEFT JOIN pdm_rating as t3 with(nolock) ON t1.pdm_cand_id=t3.pdm_cand_id  RIGHT JOIN headers as t7 with(nolock) ON t1.header_id=t7.header_id WHERE  ISNULL(t2.rank,0) BETWEEN 4 AND 5 AND t3.pdm_rating_instance_id=22 AND t3.value >= 0.5 AND t7.source_name LIKE 'G%'"}
        act_tmplt = self.palfa_sampleqry.get_active_text()
        self.palfaqrybuf.set_text(tmplts[act_tmplt])
        
    def on_qry_execute(self, event):
        """
        respond to "Execute Query" events.
        Read in username/password (prompting if not set),
        and run the query

        """
        try:
            import pymssql
        except ImportError:
            print "PALFA query requires pymssql module. Aborting query"
            return
        print "executing query"

        try:
            from config.commondb import username, password
            pwd = username 
            uname = password
        except:
            pwd = self.palfa_qu.get_text()
            uname = self.palfa_qp.get_text()

        if '' in [uname, pwd]:
            print "A username and password are required. Aborting query"
            return

        
        dbhost = Vdecode(uname+pwd, 'ugkagdipzy.nm.qigtcjn.bmh')
        dbversion = int(self.builder.get_object("dbversion").get_value())
        if dbversion == 2:
            dbv = 'JPRDYEukmQQ2'
        else:
            dbv = 'JPRDYEukmQQ3'
        print "DBV", dbv
        database = Vdecode(uname+pwd, dbv)
        table_name = Vdecode(pwd+uname, 'EXW_Qucjgbcnb_Oxhkowyh_Dgnyphfiyw')
        
        #file with list of query subs
        fin = self.palfaqry_subfile.get_filename() 
        #actual query to perform
        srt = self.palfaqrybuf.get_start_iter()
        stp = self.palfaqrybuf.get_end_iter()
        qry = self.palfaqrybuf.get_text(srt, stp,1)
        #if "by candidateid" is not active we still use that query to download the filelocation
        if qry != "SELECT * FROM PDM_Candidate_Binaries_Filesystem as t where t.pdm_cand_id = %s":
            loc_qry = "SELECT * FROM PDM_Candidate_Binaries_Filesystem as t where t.pdm_cand_id = %s"
            #(pdm_cand_bin_id,pdm_cand_id,pdm_plot_type_id,filename,file_location,uploaded
        else:
            #else we don't need to run the location query
            loc_qry = ""
#        if self.palfa_sampleqry.get_active_text() != 'by candidate id':
#            loc_qry = "SELECT * FROM PDM_Candidate_Binaries_Filesystem as t where t.pdm_cand_id = %s"
#            #(pdm_cand_bin_id,pdm_cand_id,pdm_plot_type_id,filename,file_location,uploaded
#        else:
#            #else we don't need to run the location query
#            loc_qry = ""

        try:
            db = pymssql.connect(dbhost, pwd, uname, database)
        except:
            print "Couldn't contact the database. Wrong password?"
            return 

        #perform the query if all things are set
        connection = db.cursor()
        loc_list, fn_list, png_list = palfa_query(connection, qry, fin, loc_qry)

        #keep track of the filename and location
        self.qry_results['location'] = np.array(loc_list)
        self.qry_results['filename'] = np.array(fn_list)
        self.qry_results['pngname'] = np.array([i[0] for i in png_list])
        #keep: if we vote or 'd'ownload on candidate, we keep it.
        #default is not to keep it
        self.qry_results['keep'] = np.array([False for i in png_list])

        if len(loc_list) > 0:
            self.qry_dwnld.show()
        else:
            self.qry_dwnld.hide()
        #write out the pngs to self.qrybasedir if they don't already exist
        #and create a file listing the candidates together with current voter
        if (self.qrybasedir is not None) and (not os.path.exists(self.qrybasedir)):
            os.makedirs(self.qrybasedir)
        temp = tempfile.NamedTemporaryFile(prefix='PFDqry_', suffix='.txt', \
                                               dir=self.qrybasedir,delete=False)
        if self.active_voter is None:
            #prompt for voter name
            name = inputbox('Voter chooser',\
                                'No user voters found in %s. Add your voting name')
            if name:
                act_name = name
            else:
                act_name = 'temp'
        else:
            act_name = self.voters[self.active_voter]
        temp.file.write("#%s %s\n" % ('fname', act_name))
        for pngname, pngdata in png_list:
            print "PNG",pngname
            pfdname = pngname.replace('.png','')
            temp.file.write("%s %3.4f \n"% (pfdname, np.nan))
            if os.path.exists('%s/%s' % (self.qrybasedir,pfdname)):
                continue
            f = open('%s/%s' % (self.qrybasedir,pngname),'w')
            f.write(pngdata)
            f.close()
        temp.file.flush()
        self.qry_savefil = temp.name
        #get the pfd download location
        self.qry_saveloc = self.builder.get_object('gtk_qrysaveloc').get_text()
        if not self.qry_saveloc:
            self.qry_saveloc = './'
        print "\n Will save downloaded pfd's to ", abspath(self.qry_saveloc)
        #we track which candidates we like/have-downloaded, only saving those
        self.data_fromQry = True
        #load up the data
        self.on_open(event='load', fin=self.qry_savefil)
        #reset the savefile location for future votes...
        self.savefile = None
        
    def on_qry_dwnld_clicked(self, event):
        """
        download all results from the palfa query

        """
        if self.data_fromQry:
            for fname in self.qry_results['filename']:
                if not os.path.exists(os.path.join(self.qrybasedir,fname)):
                    self.PALFA_download_qry(fname)

    def on_fl_toggle(self, widget, event=None):
        """
        respond to FL votes done by mouse-click, 
   	updating the vote

        """
        if self.active_voter is None:
            print "Please add a voter (or load a file)"
            return
        act_name = self.voters[self.active_voter] + '_FL'
        FL_votes = {0: 'FL_overall',
                    1: 'FL_profile',
                    2: 'FL_intervals',
                    3: 'FL_subbands',
                    4: 'FL_DMcurve'
                    }
        (model, pathlist) = self.pfdtree.get_selection().get_selected_rows()
#only use first selected object
        if len(pathlist) != 0:
            path = pathlist[0]
            this_iter = model.get_iter(path)
            fname = self.pfdstore[this_iter][1]

            idx = self.data['fname'] == fname
            newvote = []
            for k, v in FL_votes.iteritems():
                o = self.builder.get_object(v).get_active()
                value = np.where(o,1,0)
                newvote.append(value)
            self.data[act_name][idx] = np.array(newvote)


    def PALFA_download_qry(self, fname):
        """
        if we vote on a query candidate (just a png) or hit 'd',
        we download the candidate.

        """
        from ftplib import FTP

        #fn_name doesn't seem to be right! look into "def palfa_qry"
        tmp = np.array([i.replace('.png','') for i in self.qry_results['pngname']])
        idx = np.where(self.qry_results['filename'] == fname)[0]
        idx = np.where(tmp == fname)[0]
        if len(idx) == 1:
            self.qry_results['keep'][idx] = True
            pfdname = self.qry_results['filename'][idx]
            pfdloc = self.qry_results['location'][idx]
        elif len(idx) > 1:
            self.qry_results['keep'][idx] = True
            pfdname = self.qry_results['filename'][idx[0]]
            pfdloc = self.qry_results['location'][idx[0]]
        else:
            return 
            
        if isinstance(pfdname, str):
            pfdname = [pfdname]
        if isinstance(pfdloc, str):
            pfdloc = [pfdloc]

        try:
            from config.commondb import username, password
            pwd = username 
            uname = password
        except:
            pwd = self.palfa_qu.get_text()
            uname = self.palfa_qp.get_text()

        if '' in [uname, pwd]:
            print "A username and password are required. Aborting query"
            return
        ftp_host = Vdecode(pwd+uname, 'plozwvd.ra.wlaatfv.sxj')
        ftp_pwd = Vdecode(pwd+uname, 'CUSZ305s')
        ftp_username = Vdecode(uname+pwd, 'jprdyfuqj')
        ftp_port = 31001
        ftp = FTP()
        ftp.connect(ftp_host, ftp_port)
        ftp.login(ftp_username, ftp_pwd)
        for i,d in enumerate(pfdloc):
            ftp.cwd(d)
            filename = pfdname[i]
            savename = "%s/%s" % (self.qry_saveloc, filename)
            tmpname = abspath(os.path.join(self.qrybasedir,filename))
            if os.path.exists(savename):
                print "FTP: %s already exists" % \
                    abspath(os.path.join(self.qry_saveloc, filename))
                #make sure temp file is there:
                if not os.path.exists(tmpname):
                    os.symlink(abspath(savename), tmpname)
            else:
                sys.stdout.write("\r FTP: downloading %s... " % savename)
                ftp.retrbinary('RETR %s' % filename, open( savename, 'wb').write )
                sys.stdout.write("\r FTP: downloading %s... done" % savename)
                sys.stdout.flush()
            #create a symbolic link to this file from the temporary location (for coding-ease)
                os.symlink(abspath(savename), tmpname)
        ftp.quit()
   
        #update the tmpAI voting if necessary... and find matches
        if exists(savename) and savename.endswith('.pfd') \
                and (self.tmpAI != None) and self.tmpAI_tog.get_active():
            if savename not in self.tmpAI_avgs:
                pfd = pfdreader(savename)
                #pfd = pfddata(savename)
                #pfd.dedisperse()
                avgs = feature_predict(self.tmpAI, pfd)
                self.tmpAI_avgs[savename] = avgs
            else:
                avgs = self.tmpAI_avgs[savename]
            self.update_tmpAI_votemat(avgs)
            self.find_matches()

    def on_PALFAqry_toggled(self, event):
        """
        get status of checkbox, and display
        the palfa query window accordingly

        """
        if self.palfaqry_tog.get_active():
            self.palfaqry_win.show_all()
        else:
            self.palfaqry_win.hide()

    def remove_fname(self, fname):
        """
        hitting 'Delete' key removes this fname from the 
        list of pfd's to save
        """
        tokeep = self.data['fname'] != fname
        self.data = self.data[tokeep]
        self.dataload_update()
####### end MainFrame class ####

################################
## utilities

def palfa_query(conn, qry, fin, loc_qry):
    """
    perform the query defined in "the query" box.

    Args:
    conn : the db.connection 
    qry : the actual qry
    fin : if not '', should contain the qry substitutions
    loc_qry : if qry isn't a 'by candidate id' query which returns the pfd file location,
              then we also run the "by candidate id" query

    Returns:
    list_of_file_locations, list_of_filenames, list_of(png filenames, png data)
    """
    #recovers the png file for each candidate id.
    png_qry = 'SELECT * FROM PDM_Candidate_plots as t where t.pdm_cand_id = %s'
            #returns [(pdfm_cand_id, pdfm_plot_type_id, filename, png filedata)]

    loc_list, fn_list, png_list = [], [], []
    
    if fin: #then we require substitutions
        subs = np.genfromtxt(fin, dtype=str)
        print "Getting information for %s candidates. Be patient" % len(subs)
        for sub in subs:
            if subs.ndim == 1:
                Q = qry % sub
            else:
                Q = qry % tuple(sub)
            conn.execute(Q)
            info = conn.fetchall()
            for i in info:
                if loc_qry:
                    #recall, expect first entry of "Q" to be candid
                    conn.execute(loc_qry % i[0]) 
                    info2 = conn.fetchall()
                    #loc_qry returns (pdf_cand_bin_id, pdf_cand_id, pdfm_plot_type_id, filename, file_location,uploaded)
                    #get the actual png data
                    for j in info2:
                        filename, location = j[3], j[4]
                        location = '/' + location
                        conn.execute(png_qry % j[1]) 
                        png_info = conn.fetchall()
                        if len(png_info) > 0:
                            loc_list.append(location)
                            fn_list.append(filename)
                            png_list.append((png_info[0][2], png_info[0][3]))
                else:
                    #the original query was a 'loc_qry'-->candid=element(2)
                    filename, location = i[3], i[4]
                    location = '/' + location
                    conn.execute(png_qry % i[1])
                    png_info = conn.fetchall()
                    if len(png_info) > 0:
                        loc_list.append(location)
                        fn_list.append(filename)
                        png_list.append((png_info[0][2], png_info[0][3]))
    else:
        conn.execute(qry)
        info = conn.fetchall()
        print "Getting information for %s candidates. Be patient" % len(info)
        for n, i in enumerate(info):
            sys.stdout.write("\r  (%s/%s)" % (n+1, len(info)))
            sys.stdout.flush()
            if loc_qry:
                conn.execute(loc_qry % i[0])
                info2 = conn.fetchall()
                for j in info2:
                    filename, location = j[3], j[4]
                    location = '/' + location
                    conn.execute(png_qry % j[1])
                    png_info = conn.fetchall()
                    if len(png_info) > 0:
                        #only append if we have all the info
                        loc_list.append(location)
                        fn_list.append(filename)
                        png_list.append((png_info[0][2], png_info[0][3]))
            else:
                #the original query was a 'loc_qry'-->candid=element(2)
                filename, location = i[3], i[4]
                #print filename, location
                location = '/' + location
                loc_list.append(location)
                fn_list.append(filename)
                conn.execute(png_qry % i[1])
                png_info = conn.fetchall()
                if len(png_info) > 0:
                    #only append if we have all the info
                    loc_list.append(location)
                    fn_list.append(filename)
                    png_list.append((png_info[0][2], png_info[0][3]))
    return loc_list, fn_list, png_list

def convert(fin):
    """
    given a pfd or ps file, make the png file

    return:
    the name of the png file

    """
    global show_pfd, PFD, pyimage, opts
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
                    if os.path.exists(pin) and not os.path.exists(pout):
                        try:
                            shutil.move(pin, pout)
                        except IOError:
                            print "\n[***failed]Moving %s to %s\n" % (pin, pout)
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

    if fin.endswith('.ar2') or fin.endswith('.ar'):
        #find psrchive's pdmp executable
        if not pdmp:
            dialog = Gtk.FileChooserDialog("Locate pdmp executable", self,
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
        if pdmp:
            #make the .ps file (later converted to .png)
            pfddir = dirname(abspath(fin))
            pfdname = basename(fin)
            full_path = abspath(fin)
            cwd = os.getcwd()

            os.chdir(pfddir)
            show_name = os.path.splitext(pfdname)[0]
            #cmd = [pdmp, '-ms', '16', '-mc', '16', '-mb', '128', '-pr', '40', '-dr', '100', '-g', '%s.ps/cps' % (show_name), full_path]
            #cmd = [pdmp, '-S','-ms', '16', '-mc', '16', '-mb', '64', '-g', '%s.ps/cps' % (show_name), pfdname]
            cmd = [pdmp, '-S', '-mc', '16', '-mb', '64', '-g', '%s.ps/cps' % (show_name), pfdname]
            subprocess.call(cmd, shell=False,
                            stdout=open('/dev/null','w'))
            #subprocess.call(cmd, shell=False)
            os.chdir(cwd)

            # assign fin to ps file so it converts to png below
            fin = abspath(os.path.join(pfddir, "%s.ps" % show_name))
            #print fin, os.path.exists(fin)
        else:
            #conversion failed
            fout = None

    if fin.endswith('.spd'):
        pfddir = dirname(abspath(fin))
        pfdname = basename(fin)
        full_path = abspath(fin)
        cwd = os.getcwd()
        show_name = os.path.splitext(pfdname)[0]
        #os.system('python ' + AI_path + 'Single-pulse/show_spplots.py %s' % fin)
        show_spplots = AI_path + '/../Single-pulse/show_spplots.py'
        tgzname = pfdname.split('_')[0]+'_zerodm_singlepulse.tgz'
        if opts.spplot: #don't make the time vs DM plot, takes too long
        #if exists(tgzname):
            #print "tgz file:%s exists!" % tgzname
            cmd = ['python', show_spplots, pfdname, tgzname]
        else:
            #print "tgz file:%s does not exists!" % tgzname
            cmd = ['python', show_spplots, pfdname]
        subprocess.call(cmd, shell=False,
                            stdout=open('/dev/null','w'))
        os.chdir(cwd)
        fps = abspath(os.path.join(pfddir, "%s.ps" % pfdname))
        bdir = dirname(abspath(fps))
        bname = basename(fps)
        fout = os.path.join(bdir, bname.replace('.ps','.png')) 
    #convert to png
        if pyimage:
            f = Image(fps)
            #f.rotate(90)
            f.write(fout)
        else:
            cmd = ['convert',fps,'png:%s' % fout]
            subprocess.call(cmd, shell=False,
                            stdout=open('/dev/null','w'))


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
    read data stored in a simple txt file or numpy recarry with (at least) one column:
    filename 

    subsequent columns are added based on a users votes

    Notes:
    *We also expect the first row to be a '#' comment line with the 
    labels for each column. This must be, at least, '#fname '

    *any voter name ending with "_FL" uses the new feature-labelling ability
     this is 4 votes based on each subplot, each vote being an int 0/1
     reading of text-based _FL votes is experimental. Must be a recarray.

    *if a voter_FL column is present but not 'voter', we copy the overall 
     voter_FL vote to the 'voter' column
    """
    print "Opening %s" % fname
    has_fl_tuple = False
    if fname.endswith('.npy') or fname.endswith('.pkl'):
        data = np.load(fname)
        #make sure all single-vote entries are type 'f8'
        newdtype = []
        for d in data.dtype.descr:
            if len(d) == 2:
                name, typ = d
                if 'i' in typ:
                    print "Recasting %s from %s to float" % (name, typ)
                    newdtype.append((name,'f8'))
                else:
                    newdtype.append(d)
            elif len(d) == 3:
                name, typ, sz = d
                if not name.endswith('_FL'):
                    print "Convention is to end FL vote cols with _FL"
                    print "Renaming %s to %s" % (name, name+'_FL')
                    newdtype.append((name+'_FL',type, sz))
                else:
                    newdtype.append(d)
            else:
                newdtype.append(d)
        data = data.astype(newdtype)
    #elif fname.endswith('.txt'):
        #fin = open(fname, 'r')
        #header = fin.readline()
        #if not header.startswith('#'):raise MyError('not reading a #header!')
        #cols = header.strip('#').split()
        #namemap = {'fname':'|S200', 'Scores':float, 'scores':float, 'Score':float, 'score':float}
        #dtype = []
        #for col in cols:
            #if col in namemap:
                #dtype.append((col, namemap[col]))
            #else:
                #dtype.append((col, float))
        #data = np.loadtxt(fin, dtype=dtype)
        #fin.close()
    else:
        f = open(fname,'r')
        l1 = f.readline()
        f.close()
        if '#' not in l1:
            print "[optional] first line expected to be a comment line describing the columns"
            print "format: #fname voter1 voter2 ..."
            cols = l1.split()
            ncol = len(cols)
            if ncol > 1:
                colnames = ['fname']
                coltypes = ['|S230']
                for ni, nv in enumerate(cols[1:]):#range(ncol-1):
#                    nv = cols[ni+1]
                    try:
                        #test if this is a regular column
                        val = float(nv)
                        colnames.append('voter%s' % ni)
                        coltypes.append('f8')
                    except(TypeError,ValueError):
                        #or a feature-label column
                        coltypes.append('|S11')
                        colnames.append('voter%s_FL' % ni)
                        has_fl_tuple = True
            else:
                colnames = ['fname']
                coltypes = ['|S230']
        else:
            l1 = l1.strip('#')
            colnames = l1.split()
            #fname
            coltypes = ['|S230']
            for ni, nv in enumerate(colnames[1:]):
                if nv.endswith('_FL'):
                    coltypes.append("|S15")
                    has_fl_tuple = True
                else:
                    coltypes.append('f8')
        try:
            data = np.recfromtxt(fname, dtype={'names':colnames,'formats':coltypes},comments='#')
        except(IOError):
            data = None
            print "Couldn't parse file %s" % fname
        if data is not None:
            if has_fl_tuple:
                dtypes = data.dtype.descr
                new_dtypes = []
                for k, v in dtypes:
                    new_dtypes.append((k,v.replace('|S15','5i8')))
                new_data = np.recarray(data.size, dtype=new_dtypes)
                for k, v in dtypes:
                    if v != '|S15':
                        new_data[k] = data[k]
                    else:
                        for i in xrange(data.size):
                            new_data[k][i] = eval(data[k][i])
                data = new_data
            if 'DMCurve' in data.dtype.names:
                FLdata = data[['Overall','Profile','Interval','Subband','DMCurve']]
                data = add_voter('featurevoter_FL', data, this_dtype='5i8')
                data = add_voter('featurevoter', data)
                data['featurevoter_FL'] = FLdata.tolist()


    if data is not None:
        if len(data.dtype.names) == 1: #fname alone!
            name = inputbox('Voter chooser',\
                                'No user voters found in %s. Add your voting name' % fname)
            data = add_voter(name, data)
        #make sure the vote column is a regular (non-FL) vote. If not, 
        #copy over the 'overall' FL vote 
        if len(data.dtype.names) == 2:
            name = data.dtype.names[1]
            if name.endswith('_FL'):
                newname = name.replace('_FL','')
                print "Found feature-label voter %s. Copying overall vote to regular column %s " % (name, newname)
                data = add_voter(newname, data)
                data[newname] = data[name][:,0].astype('f8')
    return data


def add_voter(voter, data, this_dtype='f8'):
    """
    add a field 'voter' to the data array

    optional:
    this_dtype : specify the recarray dtype. 
                usually 'f8', but feature-labeling is '5i8'

    """
    if voter not in data.dtype.names:
        nrow = len(data)

        nvote = np.zeros(nrow,dtype=this_dtype)
        if this_dtype != '5i8':
            #for feature-labelling we default to "RFI",
            #but otherwise we set to nan
            nvote *= np.nan
        dtype = data.dtype.descr
        dtype.append((voter,this_dtype))

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
    if not isinstance(pfd,list):
        pfd = [pfd]
        
    features = ['phasebins', 'intervals', 'subbands', 'DMbins']
    avgs = {}
    for f in features:
        if  clf.strategy != 'adaboost':
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
    try:
        c = fractions.Fraction(a/b).limit_denominator(max_denominator=max_denom)
        return c.numerator, c.denominator
    except:
        return 0 , 0



def Vencode(key, string):
    """
    simple Vigenere cipher encoding
    inspired by:
    code.google.com/p/pysecret/source/browse/vigenere.py

    """
    l, k = [], 0 
    for i in string:
        if i.isalpha():
            if i.isupper():
                l.append( chr((ord(i) + ord(key[k])) % 26 + 65) )
            else:
                l.append( chr((ord(i) - 32 + ord(key[k])) % 26 + 97) )
        else:
            l.append(i)
        k = (k+1) % len(key)
    return "".join(l)

def Vdecode(key, string):
    """
    simple Vigenere cipher decoding
    inspired by:
    code.google.com/p/pysecret/source/browse/vigenere.py
    """
    l, k = [], 0
    for i in string:
        if i.isalpha():
            if i.isupper():
                l.append( chr((ord(i) - ord(key[k])) % 26 + 65))
            else:
                l.append(chr((ord(i) - 32 - ord(key[k])) % 26 + 97))
        else:
            l.append(i)
        k = (k+1) % len(key)
    return "".join(l)





if __name__ == '__main__':        
    parser = OptionParser()
    parser.add_option("-i", "--datafile", dest="data",
                      help="load pfd list from file FILE", metavar="FILE")
    parser.add_option("-a", "--tmpAI", dest="tmpAI",
                      help="load a temporary AI clf.pkl to gauge it's performance and see its' view of the candidates", metavar="clf.pkl")
    
    parser.add_option("-q", "--quiet",
                      action="store_false", dest="verbose", default=True,
                      help="don't print status messages to stdout")
    parser.add_option("-s", "--singlepulse",
                      action="store_true", dest="spplot", default=False,
                      help="make time vs. DM plot for single pulse candidates")

    (opts, args) = parser.parse_args()
    if len(args) > 0 and opts.data is None:
        opts.data = args[0]

    if opts.tmpAI is not None:
        if not os.path.exists(opts.tmpAI):
            opts.tmpAI = None

    #print 'opts.spplot:', opts.spplot
    app = MainFrameGTK(data=opts.data, tmpAI=opts.tmpAI, spplot=opts.spplot)    
    Gtk.main()


    

