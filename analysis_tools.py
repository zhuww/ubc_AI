"""
Some routines that may be useful in analyzing the prediction performance.

"""
import numpy as np
import pylab as plt
from itertools import cycle
from scipy import mgrid

lines = ["-",":","-.","--","_"]
#k=black, b=blue, g=green, r=red
colors = ['b', 'k', 'g', 'r']#, 'c', 'm', 'k']

def hist_overlap(A,B, idx=0, norm=True):
    """
    given two histograms, determine the overlap:
    Overlap(A,B) = \sum_i( min(A[i], B[i]) )

    args:
    A: array NumA(nbins)
    B: array NumB(nbins)
    
    Optional:
    idx: starting index of the sum "i" 
         (useful to determine the best cut (to minimize overlap))
    norm: True: normalize the Overlap with A.sum()/(A+B).sum()

    """
    k = np.array([ min(v, B[idx:][i]) for i, v in enumerate(A[idx:])])
    if norm:
        N = float(A[idx:].sum())/(A[idx:] + B[idx:]).sum()
    else:
        N = 1.
    return k.sum()*N

def plot_histogram(probs, target, title=False):
    """
    Given the list of pulsar probabilities 'proba'
    and the true target 'target', make a histogram
    of the pulsar and rfi distributions

    Assumes pulsars labelled '1', rfi '0'
 
    optional :
    title = False... add an informative title or not
    """
    if isinstance(probs, list):
        probs = np.array(probs)
    if probs.ndim == 2:
        probs = probs[...,1]

    plt.clf()
    nrfi = np.sum(target != 1)
    npulsar = np.sum(target == 1)
    npsr_clf, binpsr, patchespsr = plt.hist(probs[target==1], 25, facecolor='grey', alpha=0.65, label='pulsars (%s)' %npulsar, range=[0,1])

    nrfi_clf, binrfi_clf, patchesrfi_clf = plt.hist(probs[target!=1], 25, facecolor='w', alpha=0.65, label='RFIs (%s)' %nrfi, range=[0,1])

    plt.legend()
    plt.xlabel('AI score')
    plt.ylabel('number of candidates')

    pct, f1, prec, compl = find_best_f1(probs, target)
    overlap = hist_overlap(npsr_clf, nrfi_clf, idx=0, norm=True)
    if title:
        title = 'best (cut, P, C, f1) = (%.3f, %.3f, %.3f, %.3f), (overlap: %s)' %\
            (pct, prec, compl, f1, int(overlap))
        plt.title(title)
    plt.show()
    
def find_best_f1(proba, target):
    """
    look at different proba cuts for pulsar classification (label '1')
    to determine the best f1

    returns:
    pct of best cut, F1 of best cut, precision of best cut, completeness of best cut

    """
    if isinstance(proba, list):
        proba = np.array(proba)
    if proba.ndim == 2:
        proba = proba[...,1]

    nbins = 100
    #true pulsars
    tp = target == 1

    bestpct = 0.
    f1 = 0
    bestprec = 0.
    bestcompl = 0.
    for pct in np.linspace(0., proba.max(), nbins, endpoint=False):
        preds = np.where(proba > pct, 1, 0)
        #pulsars have value '1'
        precision = float(np.sum(preds[tp] == target[tp]))/preds.sum()
        completeness = float(np.sum(preds[tp] == target[tp]))/tp.sum()
        
        this_f1 = 2*precision*completeness/(precision+completeness)

        if this_f1 > f1:
            bestpct = pct
            f1 = this_f1
            bestprec = precision
            bestcompl = completeness
    return bestpct, f1, bestprec, bestcompl
    
def cut_performance(AIs, target, nbins=25, plot=True, norm=True, legend=True, features=None):
    """
    given a dictionary of AIs (keyword = descriptive name, value = predict_proba[...,1])
    return a dictionary of the hist_overlap as we change the %cut

    useful for determining and comparing the optimal cut, below which 
    the mixing of pulsars and rfi is too much

    Args:
    AIs: dictionary of AIs 
    target: target value
    nbins: number of phase bins to cut
    plot: True/False make the plots, or only return the data
    norm: when calculating the overlap, 
          normalize recovered fraction by by pulsar/(pulsar+rfi)
    legend: display a legend or not...
    features: a dictionary of the 'feature' for the classifier in the 'AIs' dictionary
             if 'not None', we assume target.shape = [nsamples x 5], the feature-labelled
             targets, and we get performance from that comparison.
             'feature' should be keyed by the 'AIs' keys, and value in ['phasebins', 'DMbins', 'intervals', 'subbands']

    Returns:
    1) pct cut
    2) dictionary of key=AI, val=overlap(cut)
    3) dictionary of key=AI, val=% of pulsar recovered at this cut

    """
    import pylab as plt
    from itertools import cycle
    lines = ["--","-",":","-."]
    targetmap = {'phasebins':1, 'DMbins':2, 'intervals':3, 'subbands':4, }

    if features is not None:
        assert(target.ndim == 2)

    performance = {}
    pct_recovered = {}
    for k in AIs:
        performance[k] = []
        pct_recovered[k] = []
    psr_hist = {}
    rfi_hist = {}
    for k, v in AIs.iteritems():
        #are we using feature labeling targets and classifiers?
        if features is not None:
            if features[k] in targetmap:
                label = target[:,targetmap[features[k]]]
            else:
                #then this is the overall classifier
                label = target[:,0]
        else:
            label = target[:,0]

        if v.ndim == 1:
            idcs = label == 1
            psr_hist[k] = np.histogram(v[idcs], nbins, range=[0,1])[0] #returns histogram, bin_edges
            idcs = label != 1
            rfi_hist[k] = np.histogram(v[idcs], nbins, range=[0,1])[0]
        else:
            psr_hist[k] = np.histogram(v[label==1][...,1], bins=nbins, range=[0,1])[0] #returns histogram, bin_edges
            rfi_hist[k] = np.histogram(v[label!=1][...,1], bins=nbins, range=[0,1])[0]
        
    #now change the cut and record the overlap
    pcts = []
    for i in range(nbins-1):
        pcts.append(float(i)/nbins)
        for k, v in performance.iteritems():
            A = psr_hist[k]
            B = rfi_hist[k]
            v.append( hist_overlap(A,B, idx=i, norm=norm) )
            pct_recovered[k].append(float(A[i:].sum())/A.sum())

    if plot:
        ax = plt.subplot(211)
        linecycler = cycle(lines)
        for k, v in performance.iteritems():
            ax.plot(pcts, v, next(linecycler), label=str(k))
        ax.set_xlabel('pct cut')
        ax.set_ylabel('overlap')
        if legend:
            ax.legend()

        ax = plt.subplot(212)
        linecycler = cycle(lines)
        for k, v in pct_recovered.iteritems():
            ax.plot(pcts, v, next(linecycler), label=str(k))
        ax.set_xlabel('pct cut')
        ax.set_ylabel('pulsar fraction recovered')
        if legend:
            ax.legend(loc=3)
        
        plt.show()

    return pcts, performance, pct_recovered


def plot_neuralactivity(nn, data, cls=1, std=1, imshow=False, title='', savename=None, topN=None, shift=True):
    """
    Given a neural network (assumed 1 hidden layer) and input data,
    plot the activation of the hidden neurons.
   
    Args:
    nn : a neural network
    data : a ubc_AI.training.pfddata object, 
           or the input data array 
           *Note, NN is trained with data.mean()=0, data.std() = 1.

    Optionally:
    cls = 1: use weights for this output class. Default=1
    std = 1.: only show neurons contributing more than 1-sigma of the 
             final sigmoid activation
    imshow: False (default), plot 1d 
            True, plot 2d
    title: title for plot
    savename : Default None, otherwise save to this filename
    topN : Default None, otherwise, override std. and instead show
            the 'topN' most-excited neurons
    shift : rescale the nn so it's means represents it's contribution
            to the final decision
            Default=True

    """
    import pylab as plt
    from itertools import cycle
    lines = ["--","-",":","-."]

    theta1 = nn.layers[0].theta #nfeatures+bias x nneurons
    theta2 = nn.layers[1].theta #nneurons+bias x nclass
    if not isinstance(data, type(np.array([]))):
        #assume this is pfddata object
        data = data.getdata(**nn.feature)
        
    #hidden layer output
    z1, a1 = nn.forward_propagate(data, nl=1) #nneurons, nneurons+bias
    
    #final output
    z2, a2 = nn.forward_propagate(data,nl=2) #nclass
    
    #hidden layer values, weighted by output activation for class=cls
    wall = theta2[:,cls] * a1
    #Note: P(cls) = 1/(1+exp(-wall.sum())), so wall>0 contribute to class,
    #                                          wall<0 mean you aren't in class.

    w = wall[1:] #stip off bias term

    #loop over all neurons, plotting from most-important to least, and setting
    #the mean of the neuron to the "weight" in the final activation classification
    worder = w.argsort()[::-1]

#collect all the neurons in order of importance
#(weighted by contribution to final decision)
    a = []
    for i, wi in enumerate(worder):
        weight = w[wi]
        neuron = theta1[:,wi]
        #shift the neural-pattern to have neuron.mean() = weight... 
        #so plot can indicate the significance
        if shift:
            neuron = neuron + (weight - neuron.mean())
        a.append(neuron)
    a = np.array(a)
        
#plot all neurons with > std-sigma contribution
#    plt.clf()
    plt.clf()
    fig = plt.figure(figsize=(12,9))
    if imshow:
        plt.imshow(a)
    else:
        astd = a.std()
        amean = a.mean()
        ax = plt.subplot(211)
        linecycler = cycle(lines)
        for wi, wv in enumerate(worder): #neural index (most active --> least)
            l = next(linecycler)
            n = a[wi]
        #skip non-extreme neurons
            if topN is None:
                if abs(n.mean() - amean) >= std*astd: 
                    ax.plot(n, l, label='nn%s (%0.2f,%0.2f)' % (wv,z1[wv],w[wv]))
#                    ax.plot(n, l, label='nn%s (%0.2f)' % (wv,w[wv]))
            else:
                if (wi < topN) | (wi >= len(worder) - topN):
                    ax.plot(n, l, label='nn%s (%0.2f,%0.2f)' % (wv,z1[wv],w[wv]))
#                    ax.plot(n, l, label='nn%s (%0.2f)' % (wv,w[wv]))
        ax.set_xlabel('phase bin')
        ax.set_ylabel('activation contribution')
        ax.legend(loc=7)

        ax = plt.subplot(212)
        ax.plot(data)
        ax.set_xlabel('phase bin')
        ax.set_ylabel('Intensity')
        ax.set_title(title)
    if savename is not None:
        fig.savefig(savename)
    else:
        plt.show()
    
    

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
              should be list of tuples [('label1', data1), ('label2', data2)...]

    Outputs:
      a sequence of files showing the classifiers performance as a function
      of phase shift, as well as displaying the phase-shifted data so that
      it can be visually inspected for problems.

    """
    curclass = clf.__class__
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
    x = mgrid[0:1.-1./nbin:nbin*1j]


    #get Prob(phase) first
    for shift in range(nbin):
        sdata = np.roll(data, shift, axis=D-1)
        if clf.use_pca:
            sdata = clf.pca.transform(sdata.flatten())
        if D == 1:
            preds.append(clf.predict_proba([sdata])[...,1][0])
        else:
            preds.append(clf.predict_proba([sdata.flatten()])[...,1][0])

    clfname = str(type(clf)).split('.')[-1].strip('>').strip("'")
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
                         str(type(clf)).split('.')[-1].strip('>').strip("'"))
            ax1.plot(x[shift], preds[shift], 'bo', markersize=10, alpha=0.5)
            if preds[shift] > .88:
                offset = -.05
            else:
                offset = .05
            ax1.text(x[shift], preds[shift]+offset, '%.03f' % preds[shift],
                     bbox={'facecolor':'blue', 'alpha':0.5, 'pad':10})
            if compare is not None:
                for name, data in compare:
                    cbin = len(data)
                    comp_coords = mgrid[1:1.-1./cbin:cbin*1j]
                    ax1.plot(comp_coords, data, 'r',label=name)
                    cdata = np.interp(x, comp_coords, data)
                    ax1.plot(x[shift], cdata[shift], 'ro', markersize=10, alpha=0.5)
                    if cdata[shift] > .88:
                        offset = -0.05
                    else:
                        offset = .05
                    ax1.text(x[shift], cdata[shift]+offset, '%.03f' % cdata[shift],
                             bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
            ax1.set_ylabel('Probability')
            ax1.set_title('%s, %s, shift %i' % \
                              (str(type(clf)).split('.')[-1].strip('>').strip("'"),
                               clf.feature, shift))
            ax1.set_ylim(0, 1)
            ax1.set_xlabel('Phase Shift')
            if compare is not None:
                names = [i for i,v in compare]
#                    ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#                                ncol=2, mode="expand", borderaxespad=0.)
                ax1.legend(loc='lower center',ncol=len(names), mode="expand")
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
                         str(type(clf)).split('.')[-1].strip('>').strip("'"))
            ax1.plot(x[shift], preds[shift], 'bo', markersize=10, alpha=0.5)
            if preds[shift] > .88:
                offset = -0.05
            else:
                offset = 0.05
            ax1.text(x[shift], preds[shift]+offset, '%.03f' % preds[shift],
                     bbox={'facecolor':'blue', 'alpha':0.5, 'pad':10})
            if compare is not None:
                for name, data in compare:
                    cbin = len(data)
                    comp_coords = mgrid[0:1.-1./cbin:cbin*1j]
                    cdata = np.interp(x, comp_coords, data)
                    ax1.plot(x, cdata, 'r',label=name)
                    ax1.plot(x[shift], cdata[shift], 'ro', markersize=10, alpha=0.5)
                    if cdata[shift] > .88:
                        offset = -0.05
                    else:
                        offset = 0.05
                    ax1.text(x[shift], cdata[shift]+offset, '%.03f' % cdata[shift],
                             bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
            ax1.set_ylabel('Probability')
            ax1.set_title('%s, %s, shift %i' % \
                              (str(type(clf)).split('.')[-1].strip('>').strip("'"),
                               clf.feature, shift))
            ax1.set_ylim(0,1)
            plt.setp( ax1.get_xticklabels(), visible=False)
            if compare is not None:
                names = [i for i,v in compare]
#                    ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#                               ncol=2, mode="expand", borderaxespad=0.)
                ax1.legend(loc='lower center',ncol=len(names),\
                               mode="expand")
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
