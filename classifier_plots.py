"""
Aaron Berndsen:
a collection of routines to make pretty plots for
combinedAI and ubc_AI.classifier objects.

They are primarily diagnostic plots, showing the behaviour
of the classifier as the phase is shifted.

"""
import pylab as plt
import numpy as np
from itertools import cycle
from scipy import mgrid

lines = ["-",":","-.","--","_"]
#k=black, b=blue, g=green, r=red
colors = ['b', 'k', 'g', 'r']#, 'c', 'm', 'k']

def plot_combinedAI_shiftpredict(cAI, pfd):
    """
    plot the individual and overall performance
    of the cominedAI and it's predictors

    Args:
    cAI : a combinedAI argument
    pfd : a single pfddata object

    """
    if not isinstance(pfd, type(list())):
        pfd = [pfd]

    lop = cAI.shift_predictions(pfd, True)#[nsamples, nclassifiers, nbins]
    lop = lop[0,:,:] #[nclassifiers, nbins]
    lopm = lop.mean(axis=0)
    lopp = []
    if cAI.strategy in cAI.AIonAIs: #
        for v in lop.transpose():
            lopp.append( cAI.AIonAI.predict_proba(v)[...,1][0])
    lopp = np.array(lopp)
    
    fig = plt.figure()
    ax = fig.add_subplot(221)
    nbin = lopm.shape[0]
    coords = mgrid[0:1.-1./nbin:nbin*1j]
    ax.plot(coords, lopm, label='cAI.mean vote')
    if cAI.strategy in cAI.AIonAIs:
        ax.plot(coords, lopp, label='cAI %s' % cAI.strategy)           
    ax.set_xlabel('phase shift')
    ax.set_ylabel('Probabililty')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.legend(loc='lower center', ncol=2, mode='expand')
    ax.set_title(pfd[0].filenm)


    linecycler = cycle(lines)
    colorcycler = cycle(colors)
    ax1 = fig.add_subplot(223)
    for fi, f in enumerate(cAI.list_of_AIs):
        clfname = str(type(f)).split('.')[-1].strip('>').strip("'")
        clfname = clfname.replace('NeuralNetwork', 'NN')
        clfname = clfname.replace('LogisticRegression', 'LR')
        clfname = clfname.replace('pnnclf', 'NN')
        clfname = clfname.replace('svmclf','SVC')
        lbl = "%s %s" % (clfname, f.feature)
        if f.feature.keys()[0] != 'DMbins':
            v = lop[fi]
            ax1.plot(coords, v, next(linecycler),\
                         color=next(colorcycler), label=lbl) 

    ax1.set_ylim(0, 1)
    ax1.set_xlim(0, 1)
    ax1.set_xlabel('phase shift')
    ax1.set_ylabel('Probability')
    ax1.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
    plt.show()


def plot_classifier_shiftpredict(clf, pfd, compare=None):
    """
    Plot the predictions vs phase-shift for a single ubc_AI.classifier.
    This routine generate 'nbin' snapshot files, useful for 
    seeing the data for a particular phase shift.

    Args:
      clf : the ubc_AI.classifier object
      pfd : a single pfddata file
      compare : optional Prob(phase) for other classifiers

    Outputs:
      a sequence of files showing the classifiers performance as a function
      of phase shift, as well as displaying the phase-shifted data so that
      it can be visually inspected for problems.

    """
    clf.__class__ = clf.orig_class
    
    if isinstance(pfd, type(list())):
        pfd = pfd[0]

    data = np.array(pfd.getdata(**clf.feature))

    nbin = clf.feature.values()[0]
    feature = clf.feature.keys()[0]
    if feature in 'DMbins': 
        print "DMbins doesn't have phase bins. Exiting"
        return

    if feature in ['phasebins']:
        D = 1
    else:
        D = 2
    if D == 2:
        data = data.reshape((nbin, nbin))
    preds = []
    x = mgrid[0:1:nbin*1j]

    if compare is not None:
        comp_coords = mgrid[0:1:1j*len(compare)]
        compdata = np.interp(x, comp_coords, compare)


    #get Prob(phase) first
    for shift in range(nbin):
        sdata = np.roll(data, shift, axis=D-1)
        if clf.use_pca:
            sdata = clf.pca.transform(sdata.flatten())
        if D == 1:
            preds.append(clf.predict_proba([sdata])[...,1][0])
        else:
            preds.append(clf.predict_proba([sdata.flatten()])[...,1][0])

    clfname = str(type(self)).split('.')[-1].strip('>').strip("'")
    clfname = clfname.replace('NeuralNetwork', 'NN')
    clfname = clfname.replace('LogisticRegression', 'LR')
    for shift in range(nbin):
        fout = "%s_%s%i-%03d" % (clfname, feature, nbin, shift)
        plt.clf()
        plt.subplots_adjust(hspace=0.15)
        sdata = np.roll(data, shift, axis=D-1)

        # show Prob(phase), orig data(phase), pca data(phase)
        if clf.use_pca:
            pdata = clf.pca.inverse_transform(clf.pca.transform(sdata.flatten()))

            ax1 = plt.subplot2grid((2,2), (0,0), colspan=2)#, aspect='equal')
            ax1.plot(x, preds, 'b',label='%s' % \
                         str(type(self)).split('.')[-1].strip('>').strip("'"))
            ax1.plot(x[shift], preds[shift], 'bo', markersize=10, alpha=0.5)
            if preds[shift] > .88:
                offset = -.05
            else:
                offset = .05
            ax1.text(x[shift], preds[shift]+offset, '%.03f' % preds[shift],
                     bbox={'facecolor':'blue', 'alpha':0.5, 'pad':10})
            if compare is not None:
                ax1.plot(comp_coords, compare, 'r',label='combinedAI')
                ax1.plot(x[shift], compdata[shift], 'ro', markersize=10, alpha=0.5)
                if compdata[shift] > .88:
                    offset = -0.05
                else:
                    offset = .05
                ax1.text(x[shift], compdata[shift]+offset, '%.03f' % compdata[shift],
                         bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
            ax1.set_ylabel('Probability')
            ax1.set_title('%s, %s, shift %i' % \
                              (str(type(self)).split('.')[-1].strip('>').strip("'"),
                               clf.feature, shift))
            ax1.set_ylim(0, 1)
            ax1.set_xlabel('Phase Shift')
            if compare is not None:
#                    ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#                                ncol=2, mode="expand", borderaxespad=0.)
                ax1.legend(loc='lower center',ncol=2, mode="expand")
#                    ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#                plt.setp( ax1.get_xticklabels(), visible=False)

            ax2 = plt.subplot2grid((2,2), (1,0), aspect='equal')
            if D == 2:
                ax2.imshow(sdata, cmap=plt.cm.gray)
            else:
                ax2.plot(x, sdata)
            if feature == 'phasebins':
                ax2.set_ylabel('Shifted Profile (orig)')
            else:
                ax2.set_ylabel('%s (orig)' % feature)
            ax2.set_xlabel('Phase')
            plt.setp( ax2.get_yticklabels(), visible=False)
            plt.setp( ax2.get_xticklabels(), visible=False)

            ax3 = plt.subplot2grid((2, 2), (1,1), aspect='equal')
            if D == 2:
                pdata = pdata.reshape((nbin, nbin))
                ax3.imshow(pdata, cmap=plt.cm.gray)
                ax3.set_xticks([0,nbin/4,nbin/2,3*nbin/4,nbin],\
                                   ['0','.25','.5','.75','1'])
            else:
                ax3.plot(x, pdata)
            if feature == 'phasebins':
                ax3.set_ylabel('Shifted Profile (pca)')
            else:
                ax3.set_ylabel('%s (pca)' % feature)
            plt.setp( ax3.get_xticklabels(), visible=False)
            plt.setp( ax3.get_yticklabels(), visible=False)

            ax3.set_xlabel('Phase')

        else:
            plt.subplots_adjust(hspace=0)
            ax1  = plt.subplot(2,1,1)
            ax1.plot(x, preds, 'b',label='%s' % \
                         str(type(self)).split('.')[-1].strip('>').strip("'"))
            ax1.plot(x[shift], preds[shift], 'bo', markersize=10, alpha=0.5)
            if preds[shift] > .88:
                offset = -0.05
            else:
                offset = 0.05
            ax1.text(x[shift], preds[shift]+offset, '%.03f' % preds[shift],
                     bbox={'facecolor':'blue', 'alpha':0.5, 'pad':10})
            if compare is not None:
                ax1.plot(x, compdata, 'r',label='combinedAI')
                ax1.plot(x[shift], compdata[shift], 'ro', markersize=10, alpha=0.5)
                if compdata[shift] > .88:
                    offset = -0.05
                else:
                    offset = 0.05
                ax1.text(x[shift], compdata[shift]+offset, '%.03f' % compdata[shift],
                         bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
            ax1.set_ylabel('Probability')
            ax1.set_title('%s, %s, shift %i' % \
                              (str(type(self)).split('.')[-1].strip('>').strip("'"),
                               clf.feature, shift))
            ax1.set_ylim(0,1)
            plt.setp( ax1.get_xticklabels(), visible=False)
            if compare is not None:
#                    ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#                               ncol=2, mode="expand", borderaxespad=0.)
                ax1.legend(loc='lower center',ncol=2, mode="expand")
#                    ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

            ax2 = plt.subplot(2,1,2)
            if D == 2:
                ax2.imshow(sdata, cmap=plt.cm.gray, aspect='equal')
            else:
                ax2.plot(x, sdata)
            if feature == 'phasebins':
                ax2.set_ylabel('Shifted Profile')
            else:
                ax2.set_ylabel('%s (orig)' % feature)
            ax2.set_xlabel('Phase (shift)')


        plt.savefig(fout)

    clf.__class__ = curclass
