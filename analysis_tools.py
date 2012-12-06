"""
Some routines that may be useful in analyzing the prediction performance.

"""
import numpy as np
import pylab as plt


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

def plot_histogram(probs, target, title=''):
    """
    Given the list of pulsar probabilities 'proba'
    and the true target 'target', make a histogram
    of the pulsar and rfi distributions

    Assumes pulsars labelled '1', rfi '0'
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
    #title = '%s, best (cut, P, C, f1) = (%.3f, %.3f, %.3f, %.3f), (overlap: %s)' %\
        #(title, pct, prec, compl, f1, int(overlap))

    #plt.title(title)
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
    
def cut_performance(AIs, target, nbins=25, plot=True, norm=True):
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


    Returns:
    1) pct cut
    2) dictionary of key=AI, val=overlap(cut)
    3) dictionary of key=AI, val=% of pulsar recovered at this cut

    """
    import pylab as plt
    from itertools import cycle
    lines = ["--","-",":","-."]

    performance = {}
    pct_recovered = {}
    for k in AIs:
        performance[k] = []
        pct_recovered[k] = []
    psr_hist = {}
    rfi_hist = {}
    for k, v in AIs.iteritems():
        if v.ndim == 1:
            psr_hist[k] = np.histogram(v[target==1], nbins, range=[0,1])[0] #returns histogram, bin_edges
            rfi_hist[k] = np.histogram(v[target!=1], nbins, range=[0,1])[0]
        else:
            psr_hist[k] = np.histogram(v[target==1][...,1], bins=nbins, range=[0,1])[0] #returns histogram, bin_edges
            rfi_hist[k] = np.histogram(v[target!=1][...,1], bins=nbins, range=[0,1])[0]
        
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
            ax.plot(pcts, v, next(linecycler), label=k)
        ax.set_xlabel('pct cut')
        ax.set_ylabel('overlap')
        ax.legend()

        ax = plt.subplot(212)
        linecycler = cycle(lines)
        for k, v in pct_recovered.iteritems():
            ax.plot(pcts, v, next(linecycler), label=k)
        ax.set_xlabel('pct cut')
        ax.set_ylabel('pulsar fraction recovered')
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
    
    
