from ubc_AI.prepfold import pfd
from samples import downsample, normalize
import numpy as np
import psr_utils
import matplotlib.pyplot as plt
#from scipy.linalg import svd
#from pylab import *

class pfddata(pfd):
    initialized = False
    #__counter__ = [0]
    def __init__(self, filename, align=True, centre=True):
        """
        pfddata: a wrapper class around prepfold.pfd
        
        Args:
        filename : the pfd filename, or "self", then don't try to load a file.

        Optionally: 
        align : ensure that binned data falls on max(sum profile).
                this aids in interpolation of the original data onto 
                the downsampled grid. [Default = False]
                Improved summed profile (negligible change to intervals and subband plots)
        centre : shift the feature to the phase 0.5 
                 (classifier.combinedAI.fit has a randomshift parameter which can re-randomize things)
                
        """
        if not filename == "self":
            pfd.__init__(self, filename)
        self.dedisperse(DM=self.bestdm, doppler=1)
        self.adjust_period()
        #pfddata.__counter__[0] += 1
        #print pfddata.__counter__
        #print 'file initialization No.:', pfddata.__counter__[0]
        if not 'extracted_feature' in self.__dict__:
            self.extracted_feature = {}
        self.extracted_feature.update({"ratings:['period']":np.array([self.topo_p1])})


        if centre:
            mx = self.profs.sum(0).sum(0).argmax()
            nbin = self.proflen
            #number of bins from 
            noff = nbin/2 - mx
            self.profs = np.roll(self.profs, noff, axis=-1)
        if align:
            #ensure downsampled grid falls bin of max(profile)
            self.align = self.profs.sum(0).sum(0).argmax()
        else:
            self.align = 0
        self.initialized = True
        

    def getdata(self, phasebins=0, freqbins=0, timebins=0, DMbins=0, intervals=0, subbands=0, bandpass=0, ratings=None):
        """
        input: feature=feature_size
        possible features:
            phasebins: summmed profile, data cube (self.profs) summed(projected) to the phase axis.
            freqbins: summed frequency profile, data cube projected to the frequency axis
            timebins: summed time profile, data cube projected to the time axis.
            DMbins: DM curves.
            intervals: the time vs phase image
            subbands: the subband vs phase image
            ratings: List of possible rating stored in the pfd file, possible values including: period, redchi2, offredchi2, avgvoverc
        usage examples:

        """
        if not 'extracted_feature' in self.__dict__:
            self.extracted_feature = {}
        profs = self.profs

        if not self.initialized:
            print 'pfd not initialized.'
            self.__init__('self')

        def getsumprofs(M):
            feature = '%s:%s' % ('phasebins', M)
            if M == 0:
                return np.array([])
            if not feature in self.extracted_feature:
                data = profs.sum(0).sum(0)
                self.extracted_feature[feature]  = normalize(downsample(data,M,align=self.align).ravel())
            return self.extracted_feature[feature]
        def getfreqprofs(M):
            feature = '%s:%s' % ('freqbins', M)
            if M == 0:
                return np.array([])
            if not feature in self.extracted_feature:
                self.extracted_feature[feature] = normalize(downsample(profs.sum(1).sum(0),M).ravel())
            return self.extracted_feature[feature]
        def gettimeprofs(M):
            feature = '%s:%s' % ('timebins', M)
            if M == 0:
                return np.array([])
            if not feature in self.extracted_feature:
                self.extracted_feature[feature] = normalize(downsample(profs.sum(0).sum(1),M).ravel())
            return self.extracted_feature[feature]
        def getbandpass(M):
            feature = '%s:%s' % ('bandpass', M)
            if M == 0:
                return np.array([])
            if not feature in self.extracted_feature:
                self.extracted_feature[feature] = normalize(downsample(profs.sum(0).sum(1),M).ravel())
            return self.extracted_feature[feature]
        def getDMcurve(M): # return the normalized DM curve downsampled to M points
            feature = '%s:%s' % ('DMbins', M)
            if M == 0:
                return np.array([])
            if not feature in self.extracted_feature:
                ddm = (self.dms.max() - self.dms.min())/2.
                loDM, hiDM = (self.bestdm - ddm , self.bestdm + ddm)
                loDM = max((0, loDM)) #make sure cut off at 0 DM
                hiDM = max((ddm, hiDM)) #make sure cut off at 0 DM
                N = 100
                interp = False
                sumprofs = self.profs.sum(0)
                if not interp:
                    profs = sumprofs
                else:
                    profs = np.zeros(np.shape(sumprofs), dtype='d')
                DMs = psr_utils.span(loDM, hiDM, N)
                chis = np.zeros(N, dtype='f')
                subdelays_bins = self.subdelays_bins.copy()
                for ii, DM in enumerate(DMs):
                    subdelays = psr_utils.delay_from_DM(DM, self.barysubfreqs)
                    hifreqdelay = subdelays[-1]
                    subdelays = subdelays - hifreqdelay
                    delaybins = subdelays*self.binspersec - subdelays_bins
                    if interp:
                        interp_factor = 16
                        for jj in range(self.nsub):
                            profs[jj] = psr_utils.interp_rotate(sumprofs[jj], delaybins[jj],
                                                                zoomfact=interp_factor)
                        # Note: Since the interpolation process slightly changes the values of the
                        # profs, we need to re-calculate the average profile value
                        avgprof = (profs/self.proflen).sum()
                    else:
                        new_subdelays_bins = np.floor(delaybins+0.5)
                        for jj in range(self.nsub):
                            #profs[jj] = psr_utils.rotate(profs[jj], new_subdelays_bins[jj])
                            delay_bins = int(new_subdelays_bins[jj] % len(profs[jj]))
                            if not delay_bins==0:
                                profs[jj] = np.concatenate((profs[jj][delay_bins:], profs[jj][:delay_bins]))

                        subdelays_bins += new_subdelays_bins
                        avgprof = self.avgprof
                    sumprof = profs.sum(0)
                    chis[ii] = self.calc_redchi2(prof=sumprof, avg=avgprof)
                DMcurve = normalize(downsample(chis, M))
                self.extracted_feature[feature] = DMcurve
            return self.extracted_feature[feature]

        def greyscale(img):
            global_max = np.maximum.reduce(np.maximum.reduce(img))
            min_parts = np.minimum.reduce(img, 1)
            img = (img-min_parts[:,np.newaxis])/global_max
            return img

        def getintervals(M):
            feature = '%s:%s' % ('intervals', M)
            if M == 0:
                return np.array([])
            if not feature in self.extracted_feature:
                img = greyscale(self.profs.sum(1)) 
                #U,S,V = svd(img)
                #imshow(img)
                #m,n = img.shape
                #S = resize(S,[m,1]) * eye(m,n)
                #k = 6
                #imshow(np.dot(U[:,1:k], dot(S[1:k,1:k],V[1:k,:])))
                #show()
                #if M <= len(S):
                    #return S[:M]
                #else:
                    #while len(S) < M:
                        #np.append(S, 0.)
                    #return S
                #self.extracted_feature[feature] = normalize(downsample(img, M, align=self.align).ravel())#wrong!
                self.extracted_feature[feature] = normalize(downsample(img, M, align=self.align)).ravel()
            return self.extracted_feature[feature]

        def getsubbands(M):
            feature = '%s:%s' % ('subbands', M)
            if M == 0:
                return np.array([])
            if not feature in self.extracted_feature:
                img = greyscale(self.profs.sum(0))
                #U,S,V = svd(img)
                #if M <= len(S):
                    #return S[:M]
                #else:
                    #while len(S) < M:
                        #np.append(S, 0.)
                    #return S
                #self.extracted_feature[feature] = normalize(downsample(img, M, align=self.align).ravel())
                self.extracted_feature[feature] = normalize(downsample(img, M, align=self.align)).ravel()
            return self.extracted_feature[feature]

        def getratings(L):
            feature = '%s:%s' % ('ratings', L)
            if L == None:
                return np.array([])
            if not feature in self.extracted_feature:
                result = []
                for rating in L:
                    if rating == 'period':
                        result.append(self.topo_p1)
                    elif rating == 'redchi2':
                        result.append(self.calc_redchi2())
                    elif rating == 'varprof':
                        result.append(self.calc_varprof())
                    elif rating == 'offredchi2':
                        result.append(self.estimate_offsignal_redchi2())
                    elif rating == 'avgvoverc':
                        result.append(self.avgvoverc)
                    else:
                        result.append(self.__dict__[rating])
                self.extracted_feature[feature] = np.array(result)
            return self.extracted_feature[feature]


        
        data = np.hstack((getsumprofs(phasebins), getfreqprofs(freqbins), gettimeprofs(timebins), getbandpass(bandpass), getDMcurve(DMbins), getintervals(intervals), getsubbands(subbands), getratings(ratings)))
        return data 

from random import shuffle

class cross_validation(object):
    @classmethod
    def cross_val_score(self, classifier, data, target, cv=5, verbose=False):
        #classifier = classifier()
        nclasses = len(np.unique(target))
        if verbose:cv = 1
        scores = np.array([])
        for i in range(cv):
            L = len(data)
            data = np.array(data)
            index = range(L)
 # keep shuffling until training set has all types
            while 1:
                shuffle(index)
                cut = int(0.6*L)
                training_idx = index[:cut]
                test_idx = index[cut:]

                training_data = data[training_idx]
                training_target = target[training_idx]
                test_data = data[test_idx]
                test_target = target[test_idx]
                if len(np.unique(training_target)) == len(np.unique(target)):
                    break
            n_samples = len(training_data)
            training_data = training_data.reshape((n_samples, -1))
            #classifier = svm.SVC(gamma=0.1, scale_C=False)
            classifier.fit(training_data, training_target)

            if nclasses == 2:
                F1 = singleclass_score(classifier, test_data, test_target, verbose=verbose)
            else:
                F1 = multiclass_score(classifier, test_data, test_target,
                                      nclasses = nclasses, verbose=verbose)
            scores = np.append(scores, F1)

        return scores

def multiclass_score(classifier, test_data, test_target, nclasses, aclass=None, verbose=True):
        """
        Returns the mean F1 score on the given test data and labels
        or, if specified, the F1 score for a particular target 'aclass'
    
        Parameters
        ----------
        classifier : the classifier (must have .predict and .fit routines)

        test_data : array-like, shape = [n_samples, n_features]
        Training set.
        
        test_target : array-like, shape = [n_samples]
        Labels for test_data.

        nclasses : number of target classifications
        
        aclass = None : if specified, return the F1 score for this class, not the total 
        Returns
        -------
        z : float (= average F1 score)

        """
        pred_cls = {}
        true_cls = {}
        for cls in range(nclasses):
            pred_cls[cls] = set([])
            true_cls[cls] = set([])

#sort the prediction and true classes            
        for i, s in enumerate(test_data):
            predict = classifier.predict(s)[0]
            true_cls[test_target[i]].add(i)
            pred_cls[predict].add(i)

# a one vs all F1 test
        tot_accuracy = 0.
        tot_F1 = 0.
        aF1 = 0
        for k in range(nclasses):
            hit = pred_cls[k] & true_cls[k]
            miss = pred_cls[k] - true_cls[k]
            falsepos = true_cls[k] - pred_cls[k] 
            precision = np.divide(float(len(hit)), len(pred_cls[k]))
            recall = np.divide(float(len(hit)), len(true_cls[k]))
            accuracy = (np.divide(float(len(hit)), len(true_cls[k])) * 100)
            tot_accuracy += accuracy
            #F1 = 2/((1./precision)+(1./recall))
            F1 = 2 * precision * recall / (precision + recall)
            if aclass != None:
                if k == aclass:
                    aF1 = F1
            tot_F1 += F1
            if verbose:
                print "\nClass %s:" % k
                print 'accuracy: ', '%.0f%%' % (np.divide(float(len(hit)),len(true_cls[k])) * 100)
                print 'miss: ', '%.0f%%' % (np.divide(float(len(miss)),len(true_cls[k])) * 100)
                print 'false positives: ', '%.0f%%' % (np.divide(float(len(falsepos)),len(pred_cls[k]))* 100)
                print 'precision: ', '%.0f%%' % (precision* 100)
                print 'recall: ', '%.0f%%' % (recall* 100)

        tot_accuracy = tot_accuracy / nclasses
        tot_F1 = tot_F1 / nclasses
        if aclass == None:
            return tot_F1
        else:
#            print "returngin F1 score for class", aclass
            return aF1

def singleclass_score(classifier, test_data, test_target, verbose=False):
    pulsar = set([])
    truepulsar = set([])
    pred = []
    for i,s in enumerate(test_data):
        predict = classifier.predict(s)[0]
        pred.append(predict)
        #print test_target[i], int(predict)
        if int(test_target[i]) == 1:
            truepulsar.add(i)
        if  int(predict) == 1:
            pulsar.add(i)

    hit = pulsar & truepulsar
    miss = truepulsar - pulsar
    falsepos = pulsar - truepulsar
    precision = np.divide(float(len(hit)),len(pulsar))
    recall = np.divide(float(len(hit)),len(truepulsar))
    #F1 = 2/((1./precision)+(1./recall))
    F1 = 2 * precision * recall / (precision + recall)
    pred = np.array(pred)
    #print np.mean(np.where(pred == test_target, 1, 0))
    if verbose:
        print 'accuracy: ', '%.0f%%' % (np.divide(float(len(hit)),len(truepulsar)) * 100)
        print 'miss: ', '%.0f%%' % (np.divide(float(len(miss)),len(truepulsar)) * 100)
        print 'false positives: ', '%.0f%%' % (np.divide(float(len(falsepos)),len(pulsar))* 100)
        print 'precision: ', '%.0f%%' % (precision* 100)
        print 'recall: ', '%.0f%%' % (recall* 100)
        print 'F1: ', F1
   
    return F1


def learning_curve(classifier, X, y,
                   Xval=None, 
                   yval=None,
                   gamma=None,
                   pct=0.6,
                   plot=False):
    """
    returns the training and cross validation set errors
    for a learning curve
    (good for exploring effect of number of training samples)
    
    Args:
    X : training data
    y : training value
    Xval : test data
    yval : test value
    gamma : default None uses the objects value,
           otherwise gamma=0.
    pct (0<pct<1) : split the data as "pct" training, 1-pct testing
                   only if Xval = None
                   default pct = 0.6
    plot : False/[True] optionally plot the learning curve
    
    Note: if Xval == None, then we assume (X,y) is the entire set of data,
          and we split them up using split_data(data,target)

    returns three vectors of length(ntrials):
    error_train : training error for the N=length(pct*X) 
    error_val : error on x-val data, when trainined on "i" samples
    ntrials

    error = cost_Function(lambda=0)

    notes:
    * a high error indicates lots of bias,
      that you are probably underfitting the problem
      (so add more neurons/layers, or lower regularization)
     
    * for lots of trials, a high gap between training_error
      and test_error (x-val error) indicates lots of variance
      (you are over-fitting, so remove some neurons/layers,
       or increase the regularization parameter)

    """
    if not Xval:
        X, y, Xval, yval = split_data(X, y, pct=pct)

    #if gamma == None:
        #gamma = classifier.gamma

    m = X.shape[0]
#need at least one training item...
    stepsize = max(m/25,1)
    ntrials = range(1,m,stepsize)
    mm = len(ntrials)
    t_error = np.zeros(mm)
    v_error = np.zeros(mm)
    for i, v in enumerate(ntrials):
        #fit with regularization
        classifier.fit(X[0:v+1], y[0:v+1])
        
        # but compute error without regularization
        t_error[i] = 1 - classifier.score(X[0:v+1], y[0:v+1])
        # use entire x-val set
        v_error[i] = 1 - classifier.score(Xval, yval)
        
    if plot:
        plt.plot(ntrials, t_error, 'r+', label='training')
        plt.plot(ntrials, v_error, 'bx', label='x-val')
        plt.xlabel('training set size')
        plt.ylabel('error')
        plt.legend()
        plt.show()

    return t_error, v_error, ntrials

def validation_curve(classifier, X, y,
                     Xval=None, 
                     yval=None,
                     gammas=None,
                     pct=0.6,
                     plot=False):
    """
    use a cross-validation set to evaluate various regularization 
    parameters (gamma)
    
    specifically:
    train the NN, then loop over a range of regularization parameters
    and select best 'gamma' (=min(costFunction(cross-val data))

    Args:
    X : training data
    y : training value
    Xval : test data
    yval : test value
    pct (0<pct<1) : if Xval=None, split into 'pct' training
                   "1-pct" testing
    gammas : a *list* of regularization values to sample
            default None uses
            [0., 0.0001, 0.0005, 0.001, 0.05, 0.1, .5, 1, 1.5, 15]
    plot : False/[True] optionally plot the validation cure
    
    Note: if Xval == None, then we assume (X,y) is the entire set of data,
          and we split them up using split_data(data,target)

    returns:
    train_error(gamma), cross_val_error(gamma), gamma, best_gamma

    """
    if not Xval:
        X, y, Xval, yval = split_data(X, y, pct)
    
    if not gammas:
        gammas = [0., 0.0001, 0.0005, 0.001, 0.05, 0.1, .5, 1., 1.5, 15.]
    
    train_error = np.zeros(len(gammas))
    xval_error = np.zeros(len(gammas))
    for gi, gv in enumerate(gammas):
        classifier.fit(X, y, gamma=gv)
        
        train_error[gi] = classifier.costFunctionU(X, y, gamma=gv)
        xval_error[gi] = classifier.costFunctionU(Xval, yval, gamma=gv)

    if plot:
        plt.plot(gammas, train_error, label='Train')
        plt.plot(gammas, xval_error, label='Cross Validation')
        plt.xlabel('gamma')
        plt.ylabel('F1')
        plt.legend()
        plt.show()

    return train_error, xval_error, gammas, gammas[xval_error.argmin()]

def split_data(data, target, pct=0.6):
    """
    Given some complete set of data and their targets,
    split the indices into 'pct' training, '1-pct' cross-vals
    
    Args:
    data = input data
    target = data classifications
    pct = 0 < pct < 1, default 0.6

    returns:
    training_data, training_target, test_data, test_target

    """
    from random import shuffle
    if isinstance(data,type([])):
        data = np.array(data)

    L = len(target)
    index = range(L)
    cut = int(pct*L)
    while 1:
        shuffle(index)
        training_idx = index[:cut]
        training_target = target[training_idx]
        training_data = data[training_idx]

        test_idx = index[cut:]
        test_target = target[test_idx]
        test_data = data[test_idx]
        
# make sure training has samples from all classes
        if len(np.unique(training_target)) == len(np.unique(target)):
            break

    return training_data, training_target, test_data, test_target

from scipy import mgrid
def feature_curve(classifier, 
                  feature,
                  originaldata,
                  bounds=None,
                  Npts=10,
                  plot=False,
                  pct=0.4):
    """
    returns the training and cross validation set errors
    for a range of sizes of a given feature.
    (good for diagnosing overfitting or underfitting/
                          high bias or high variance)
    
    Args:
    classifier:  classifier
    feature : string, name of the given feature 
    (e.g. phasebins, intervals) 
    originaldata : the original data loaded from pickled file, have ['pfds']
    and ['target']
    bounds : the range of feature sizes to explore
    plot : whether or not to plot the scores
    Npts : plot Npts points 
    pct (0<pct<1) : split the data as "pct" training, 1-pct testing
                   only if Xval = None
                   default pct = 0.6
    plot : False/[True] optionally plot the learning curve
    
    Note: if Xval == None, then we assume (X,y) is the entire set of data,
          and we split them up using split_data(data,target)

    returns three vectors of length(ntrials):
    train_score: training error for the N=length(pct*X) 
    test_score: error on x-val data, when trainined on "i" samples
    ntrials

    vals = values of feature sizes (e.g. 8 -- 32)

    notes:
    * a high error indicates lots of bias,
      that you are probably underfitting the problem
      (so add more neurons/layers, or lower regularization)
     
    * for lots of trials, a high gap between training_error
      and test_error (x-val error) indicates lots of variance
      (you are over-fitting, so remove some neurons/layers,
       or increase the regularization parameter)

    """
    pfds = originaldata['pfds']
    orig_target = originaldata['target']
    classmap = {0:[4,5], 1:[6,7]}
    target = orig_target[:]
    for k, v in classmap.iteritems():
        for val in v:
            target[orig_target == val] = k 
    if bounds == None:
        vals = mgrid[8:32:1j*Npts]
    else:
        vals = mgrid[bounds[0]:bounds[1]:1j*Npts]

    #kws = {'phasebins':0}
    kws = {}
    train_score = np.zeros_like(vals)
    test_score = np.zeros_like(vals)
    for i, val in enumerate(vals):
        kws[feature] = int(val)
        data = [pf.getdata(**kws) for pf in pfds]
        train_data, train_target, test_data, test_target = split_data(data,target, pct=pct)
        classifier.fit(train_data,train_target)
        train_score[i] = 1-classifier.score(train_data, train_target)
        test_score[i] = 1-classifier.score(test_data, test_target)
    if plot:
        plt.plot(vals, train_score, 'r+', label='training')
        plt.plot(vals, test_score, 'bx', label='x-val')
        plt.xlabel(feature)
        plt.ylabel('error')
        plt.legend()
        plt.show()
    return train_score, test_score, vals, vals[test_score.argmin()]

import cPickle
class datafitter(object):
    """
    A class to hold the data and provide methods for testing AIs.

    """

    def __init__(self, filename, classmap=None):
        """
        initialize from a filename, to create a Datafitter instance 
        that holds the data and perform fitting using provided classifier
        """
        self.trainclassifiers = {}
        with open(filename, 'r') as fileobj:
            originaldata = cPickle.load(fileobj)
            self.pfds = originaldata['pfds']
            self.orig_target = originaldata['target']
            if classmap == None:
                self.classmap = {0:[4,5], 1:[6,7]}
            else:
                self.classmap = classmap
            self.target = self.orig_target[:]
            for k, v in self.classmap.iteritems():
                for val in v:
                    self.target[self.orig_target == val] = k 

    def update_classmap(self,classmap):
        """
        update the target mapping
        Args: 
        classmap: dictionary mapping target values to key values
              Eg. classmap = {0:[4,5], 1:[6,7]} maps target 4 and 5 to '0'
        """
        self.target = self.orig_target[:]
        self.classmap = classmap
        for k, v in self.classmap.iteritems():
            for val in v:
                self.target[self.orig_target == val] = k
 
    def prepare_data(self, **kwds):
        """
        input: the keywords for pfddata class's getdata method
        output: [pf.getdata(**kwds) for pf in self.pfds]
        """
        self.kwds = kwds
        self.data = [pf.getdata(**kwds) for pf in self.pfds]
        self.split()
        #return self.data

    def split(self, pct=0.6):
        train_data, train_target, test_data, test_target = split_data(self.data, self.target, pct=pct)
        self.train_data = train_data
        self.train_target = train_target
        self.test_data = test_data
        self.test_target = test_target
# reset which classifiers have been trained 
        self.trainclassifiers = {}


    def cross_val_score(self, classifier, cv=10, verbose=False):
        #L = len(self.data[0])
        #classifier = clsFunc(L)
        scores = cross_validation.cross_val_score(classifier, self.data, self.target, cv=cv)
        print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
        return scores

    def learning_curve(self, classifier,
                       Xval=None, 
                       yval=None,
                       gamma=None,
                       pct=0.6,
                       plot=True):
        if not 'test_data' in self.__dict__ or not 'test_target' in self.__dict__:
            self.train(classifier)
        return learning_curve(classifier, self.data, self.target, Xval=Xval, yval=yval, gamma=gamma, pct=pct, plot=plot)

    def feature_curve(self, classifier, 
                      feature,
                      bounds=None,
                      Npts=10,
                      plot=True,
                      pct=0.4):
        self.kwds = {}
        self.kwds[feature] = bounds[1]
        return feature_curve(classifier, feature, {'pfds':self.pfds, 'target':self.target}, bounds=bounds, Npts=Npts, pct=pct, plot=plot)
        
    def train(self, clf):
        if not 'test_data' in self.__dict__ or not 'test_target' in self.__dict__:
            self.split()
        self.trainclassifiers[clf] = True 
        clf.fit(self.train_data, self.train_target)

    def plot_prediction(self, clf, what):
        if not 'test_data' in self.__dict__ or not 'test_target' in self.__dict__:
            self.train(clf)
        elif not clf in self.trainclassifiers:
            clf.fit(self.train_data, self.train_target)

        pdts =  clf.predict(self.test_data)
        truepulsar = set([])
        pulsar = set([])
        for i,p in enumerate(pdts):
            if int(self.test_target[i]) == 1:
                truepulsar.add(i)
            if  int(p) == 1:
                pulsar.add(i)
        hit = pulsar & truepulsar
        miss = truepulsar - pulsar
        falsepos = pulsar - truepulsar
        precision = np.divide(float(len(hit)),len(pulsar))
        recall = np.divide(float(len(hit)),len(truepulsar))
        F1 = 2 * precision * recall / (precision + recall)
        print 'accuracy: ', '%.0f%%' % (np.divide(float(len(hit)),len(truepulsar)) * 100)
        print 'miss: ', '%.0f%%' % (np.divide(float(len(miss)),len(truepulsar)) * 100)
        print 'false positives: ', '%.0f%%' % (np.divide(float(len(falsepos)),len(pulsar))* 100)
        print 'precision: ', '%.0f%%' % (precision* 100)
        print 'recall: ', '%.0f%%' % (recall* 100)
        print 'F1: ', F1

        if what in ['miss', 'falsepos', 'truepulsar', 'pulsar']:
            what = list(locals()[what])
        else:
            what = list(miss)
        plt.figure(figsize=(8,8))
        i = 0
        axisNum = 0
        ncol = min(8,int(np.sqrt(len(what))))
        nrow = ncol
        if nrow*ncol < len(what):
            nrow += 1
        
        for row in range(nrow):
            for col in range(ncol):
                axisNum += 1
                ax = plt.subplot(nrow, ncol, axisNum)
                try:
                    feature = [k for k in sorted(self.kwds, key=lambda x:self.kwds.get(x), reverse=True)][0]
                    if feature in ['intervals', 'subbands']:
                        N = self.kwds[feature]
                        ax.imshow(self.test_data[what[i]].reshape(N,N))
                    else:
                        ax.plot(self.test_data[what[i]])
                except IndexError:pass
                i += 1
                ax.set_yticklabels([])
                ax.set_xticklabels([])

        plt.show()

    def plot_samples(self, sample_list=[]):
        """
        plot the list of 'test_data' samples, given a list of
        their index numbers

        Args:
        sample_list = list of sample indices to plot (maximum 64)

        """
        if isinstance(sample_list,type(set([]))):
            sample_list = list(sample_list)
        plt.figure(figsize=(8,8))
        axisNum = 0
        ncol = min(8,int(np.sqrt(len(sample_list))))
        nrow = ncol
        if nrow*ncol < len(sample_list):
            nrow += 1
        i = 0
        for row in range(nrow):
            for col in range(ncol):
                axisNum += 1
                ax = plt.subplot(nrow, ncol, axisNum)
                try:
                    feature = [k for k in sorted(self.kwds, key=lambda x:self.kwds.get(x), reverse=True)][0]
                    if feature in ['intervals', 'subbands']:
                        N = self.kwds[feature]
                        ax.imshow(self.test_data[sample_list[i]].reshape(N,N),
                                  cmap=plt.cmap.gray)
                    else:
                        ax.plot(self.test_data[sample_list[i]])
                except IndexError:pass
                i += 1
                ax.set_yticklabels([])
                ax.set_xticklabels([])

        plt.show()


    def classifier_comparison(self, classifiers=[], true_miss=True):
        """
        given a list of classifiers, train them if necessary, 
        calculate their predictions on the self.target_data,
        and return the overlap in the classifications prediction
        
        Args:
        classifiers : list of classifiers
        true_miss :  default (T: only return objects that are, indeed, pulsars)
                     (F: return all objects predicted to be pulsars)
        returns (index of):
        "intersection of all pulsars", "union of all pulsars"
        
        if the object is, indeed, a pulsar

        Note: assume pulsar is classed/targetted as '1'
        """
        if not 'test_data' in self.__dict__ or not 'test_target' in self.__dict__:
            self.split()

#intersection of all pulsars
        ipulsar = set([])
#union of all pulsars
        upulsar = set([])
        for cli, clf in enumerate(classifiers):
            if clf not in self.trainclassifiers:
                self.train(clf)
#                clf.fit(self.train_data, self.train_target)
            p = clf.predict(self.test_data)
            if cli == 0:
                ipulsar = set(np.where(p == 1)[0])
            else:
                ipulsar = ipulsar.intersection(np.where(p == 1)[0])
            upulsar = upulsar.union(np.where(p == 1)[0])

# only keep the true pulsars
        if true_miss:
            true_pulsars = set(np.where(self.test_target == 1)[0])
            ipulsar = ipulsar.intersection(true_pulsars)
            upulsar = upulsar.intersection(true_pulsars)

        return ipulsar, upulsar
