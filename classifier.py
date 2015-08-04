import numpy.random as random
import numpy as np
from sklearn.decomposition import RandomizedPCA as PCA
from sklearn import svm, linear_model, tree, ensemble
from sklearn.ensemble import GradientBoostingClassifier as GBC

from ubc_AI.training import split_data
from ubc_AI import pulsar_nnetwork as pnn 
from ubc_AI import sktheano_cnn as skcnn

#multiprocess only works in non-interactive mode:
from ubc_AI.threadit import threadit
import multiprocessing as MP
import __main__ as MAIN
if hasattr(MAIN, '__file__'):
    InteractivePy = False
    #print "Yeah!!! we are running with multiprocessing!"
else:
    print "running in interactive python mode, multiprocessing disabled"
    InteractivePy = True

num_workers = max(1, MP.cpu_count() - 1)
if num_workers == 1: InteractivePy = True
equaleval = "%s"

class combinedAI(object):
    """
    A class to combine different AIs, and have them operate as one
    """
    def __init__(self, list_of_AIs, strategy='lr', nvote=None, score_mapper=equaleval, **kwds):
        """
        inputs
        list_of_AIs: list of classifiers
        strategy: What to do with the prediction matrix from the list_of_AIs.
                One of ['vote', 'lr', 'svm', 'forest', 'tree', 'nn', 'adaboost', 'gbc', 'kitchensink']
                Default = 'vote'
        *score_map: has to be a string that eval(score_map % score) to a function that converts the calculated probability to a new score.
        
        Notes:
        *'vote': **assumes** pulsars are labelled class 1, 
                    requires 'nvote' argument too (the number of votes to be considered a pulsar)
        *'adaboost': implementation of http://en.wikipedia.org/wiki/Adaboost
                    *only works for 2-class systems
                    *predict_proba output is not too good (arxiv.org/pdf/1207.1403.pdf)
        *'lr': uses LogisticRegression on the prediction matrix from list_of_AIs,
               and makes final prediction from the lr(predictions)
        *'svm': uses SVM on the prediction matrix from the list_of_AIs,
                and makes the final prediciton from SVM(predictions)
        *'forest': uses sklearn.ensemble.RandomForestClassifier
        *'tree': DecisionTreeClassifier
        *'nn': uses a 1-layer, N/2-neuron classifier [N=len(list_of_AIs)]
        *'gbc': use sklearn.ensemble.GraidentBoostingClassifier 
        *'kitchensink': runs SVM, LR, tree, *and* NN on prediction matrix, 
                        then takes majority vote or 'lr' for final classification

        *if strategy='vote' and nvote > 0 , nvote < len(list_of_AIs)

        """
        #things that require a 'fit'
        self.AIonAIs = ['lr', 'svm', 'forest', 'tree', 'nn', 'adaboost', 'gbc', 'kitchensink']
        #things that train on 'predict' instead of 'predict_proba'
        self.req_predict = ['adaboost', 'gbc']

        self.list_of_AIs = list_of_AIs
        self.strategy = strategy
        if strategy != 'vote' and strategy not in self.AIonAIs:
            note = "strategy %s is not recognized" % strategy
            raise MyError(note)
        if strategy == 'lr':
            self.AIonAI = linear_model.LogisticRegression(**kwds)
        elif strategy == 'svm':
            self.AIonAI = svm.SVC(probability=True, **kwds)
        elif strategy == 'forest':
            nleafs = len(list_of_AIs)/2
            self.AIonAI = ensemble.RandomForestClassifier(**kwds)
        elif strategy == 'tree':
            nleafs = len(list_of_AIs)/2
            self.AIonAI = tree.DecisionTreeClassifier(min_samples_leaf=nleafs,**kwds)
        elif strategy == 'nn':
            if 'design' in kwds:
                self.AIonAI = pnn.NeuralNetwork(**kwds)
            else:
                n = max(1,int(len(list_of_AIs)/2))
                self.AIonAI = pnn.NeuralNetwork(design=[n,2], **kwds)
        elif strategy == 'vote':
            assert( (nvote > 0) & (nvote <= len(self.list_of_AIs)) ) 
            self.nvote = nvote
        elif strategy == 'adaboost':
            self.AIonAI = adaboost(**kwds)
        elif strategy == 'gbc':
            self.AIonAI = GBC(**kwds)
        elif strategy == 'kitchensink':
            lr = linear_model.LogisticRegression(C=0.5, penalty='l1') 
            nn = pnn.NeuralNetwork(design=[64], gamma=1.5, maxiter=200) #2-class, 9-vote optimized
            svc = svm.SVC(C=15, kernel='poly', degree=5, probability=True) #grid-searched
            dtree = tree.DecisionTreeClassifier()
#            self.AIonAI = combinedAI([lr,nn,svc, dtree], nvote=2) #majority vote
#            self.AIonAI = combinedAI([lr,nn,svc, dtree], strategy='lr')
            self.AIonAI = combinedAI([lr,nn,svc,dtree], strategy='adaboost')

        self.nclasses = None #keep track of number of classes (determined in 'fit')
        self.score_mapper = score_mapper

        #initialize a feature list

    def fit(self, pfds, target, **kwds):
        """
        args: [list of pfd instances], target

        Notes:
        following advice from http://en.wikipedia.org/wiki/Ensemble_learning
        we train each classifier on a subset of the training data
        
        """
        if target.ndim == 1:
            psrtarget = target
        else:
            psrtarget = target[...,0]
        if not InteractivePy:
            #extract pfd features beforehand
            extractfeatures(self.list_of_AIs, pfds)


        input_data = []
        for n, clf in enumerate(self.list_of_AIs):
            tr_pfds, tr_target, te_pfds, te_target = split_data(pfds, target, pct=0.75)
            if InteractivePy:
                clf.fit(tr_pfds, tr_target, **kwds)
            else:
                input_data.append([clf, tr_pfds, tr_target, kwds])
        def threadfit(clf, tr_pfds, tr_target, kwds):
            clf.fit(tr_pfds, tr_target, **kwds)
            return clf
        
        if not InteractivePy:
            resultdict = threadit(threadfit, input_data)

            for n, clf in resultdict.iteritems():
                self.list_of_AIs[n] = clf

        self.nclasses = len(np.unique(target))
        if self.nclasses > 2 and self.strategy == 'adaboost':
            print "Warning, adaboost only works in 2-class systems"
            print "Reverting to Logistic Regression on the prediction matrix"
            self.strategy = 'lr'
            self.AIonAI = linear_model.LogisticRegression(penalty='l1')

        #train the AIonAI if used
        if (self.strategy in self.AIonAIs):
            if self.strategy not in self.req_predict:
                #use predict_prob 
                if InteractivePy or (len(pfds) < 5*num_workers):
                    predictions = np.hstack([clf.predict_proba(pfds)\
                                                 for clf in self.list_of_AIs]) #nsamples x (npred x nclasses)
                    #print predictions.shape
                else:
                    predictions = threadpredict_proba(self.list_of_AIs, pfds)
            else:
                #use predict
                if InteractivePy or (len(pfds) < 5*num_workers):
                    predictions = np.transpose([clf.predict(pfds)\
                                                    for clf in self.list_of_AIs]) #nsamples x npred
                else:
                    predictions = threadpredict(self.list_of_AIs, pfds)

            predictions = np.array(predictions) #nsamples x npred
            self.AIonAI.fit(predictions, psrtarget)
            

    def predict(self, pfds, pred_mat=False ):
        """
        args: 
          pfds : list of pfddata objects
        
        optionally: pred_mat = if True returns the [nsamples x npredictions] matrix
                               so you can run your own prediction combo schemes
                               (default False)
        returns:
        array of [nsamples], giving label of most-likely class

        """
        if not type(pfds) in [list, np.ndarray]:
            print "warniing: changing pfds from type %s to list" % (type(pfds))
            pfds = [pfds]

        if not InteractivePy:
            #extract pfd features beforehand
            extractfeatures(self.list_of_AIs, pfds)

        if (self.strategy in self.AIonAIs) and self.strategy not in self.req_predict:
            #use predict_proba for AI_on_AI classifier, 
            if InteractivePy or (len(pfds) < 5*num_workers):
                list_of_predicts = np.hstack([clf.predict_proba(pfds)\
                                                  for clf in self.list_of_AIs])#nsamples x (npred x classes)
            else:
                print '@Must turn off threadpredict_proba to prevent dead loop. Test not to'
                threadit.func_defaults[0]['state'] = True
                list_of_predicts = threadpredict_proba(self.list_of_AIs, pfds)
                threadit.func_defaults[0]['state'] = False
        else:
            if InteractivePy or (len(pfds) < 5*num_workers):
                list_of_predicts = np.transpose([clf.predict(pfds)\
                                                     for clf in self.list_of_AIs]) #nsamples x npred
            else:
                list_of_predicts = threadpredict(self.list_of_AIs, pfds)

        self.list_of_predicts = list_of_predicts

        self.predictions = []
        if self.strategy == 'vote':
            # return pulsar class ('1') if number of votes > nvotes
            # otherwise return the most-voted non-pulsar class
            #**assumes pulsar class is '1'

            # find N(votes)/class
            nvotes_pc = np.hstack([[np.sum(self.list_of_predicts==k,axis=1)\
                                     for k in range(self.nclasses)]]).transpose() #[nsamples x nclasses]
            npc = range(self.nclasses)[2:]
            npc.insert(0,0)
            most_votes_nonpulsar = np.argmax(nvotes_pc[:,npc], axis=1) 
            #add 1 for the missing class '1'=pulsar
            most_votes_nonpulsar[most_votes_nonpulsar != 0] += 1#[nsamples] (value =2nd  best class)

            #return pulsar if more than self.nvote votes, 
            #otherwise return most-likely non-pulsar class
            self.predictions = np.where( nvotes_pc[:,1] >= self.nvote, 1, most_votes_nonpulsar)

        elif self.strategy in self.AIonAIs:
            self.predictions = self.AIonAI.predict(self.list_of_predicts)
         
        #return np.array(self.predictions)
        if pred_mat:
            return self.list_of_predicts #if AIonAI [nsamples x (npredictions x nclasses)]
                                         #else      [nsamples x npredictions]
        else:
            return self.predictions

    def predict_proba(self, pfds):
        """
        predict_proba(self, pfds) classifier method
        Compute the likehoods each possible outcomes of the input samples.
    
        The model need to have probability information computed at training
        time: fit with attribute `probability` set to True.
    
        Parameters
        ----------
        pfds : list of pfddata objects [n_samples]

        Returns
        -------
        Returns array of [n_samples x nclasses], the probability of being in each class
        
        Notes
        -----
        * for NN, return the activation of the 'label' neuron


        """
        if not type(pfds) in [list, np.ndarray]:
            pfds = [pfds]        

        if not InteractivePy:
            #extract pfd features beforehand
            extractfeatures(self.list_of_AIs, pfds)

        if self.strategy not in self.AIonAIs:
            result = np.array([clf.predict_proba(pfds)\
                                 for clf in self.list_of_AIs]) #npreds x nsamples x nclasses
            result = result.mean(axis=0) #nsamples x nclasses
            
        else:
            #note: adaboost.predict_proba now accepts predict_proba inputs
            if self.strategy in self.req_predict and self.strategy != 'adaboost':
                if InteractivePy or (len(pfds) < 5*num_workers):
                    predicts = np.transpose([clf.predict(pfds)\
                                                 for clf in self.list_of_AIs]) #nsamples x nclasses
                else:
                    predicts = threadpredict(self.list_of_AIs, pfds)
            else:
                if InteractivePy or (len(pfds) < 5*num_workers):
                    #print 'No need to thread predict_proba (%s/%s)' % (len(pfds), 5*num_workers)#confirmed
                    predicts = np.hstack([clf.predict_proba(pfds)\
                                              for clf in self.list_of_AIs]) #nsamples x (npreds x nclasses)
                else:
                    predicts = threadpredict_proba(self.list_of_AIs, pfds)

            result = self.AIonAI.predict_proba(predicts) #nsamples x nclasses

        #renderer = lambda x:(1-x, x)
        #return np.array([res if res[1] == 0. else renderer(eval(self.score_mapper % res[1])) for res in result])
        return result

    def report_score(self, pfds, dist='PALFA_Priordists.pkl'):
        if not type(pfds) in (list,tuple):
            pfds = [pfds]

        if not self.__dict__.has_key('prior_freq_dist'):
            import cPickle
            import ubc_AI
            ubcAI_path = ubc_AI.__path__[0]
            # Note: we expect a dictionary whose key is 'Pfr_over_Pfp'
            self.prior_freq_dist = cPickle.load(open(ubcAI_path + '/' + dist, 'rb'))

        def getp0(pfd):
            #pfd.__init__('self')
            return pfd.getdata(ratings=['period'])

        def adjustscore(score, freq, w=1., spk=1.):
            """
            Apply the bayesian prior.
            w = 1., extra weight on priors, 100 is optimal.
            spk = 1., enhancement to spikes in distribution, 1.75 is optimal

            """
            newscore = []
            try:
                # the histogram (P(F0|r)/P(F0|p), bins):
                Pfr = self.prior_freq_dist['Pfr_over_Pfp']
                bin_edges = Pfr[1]
                have_prior = True
            except(KeyError):
                have_prior = False

            for i in range(len(score)):
                pp = score[i]
                f = freq[i]
                if have_prior and f > 1.:
                #if have_prior:
                    bidx = min(np.argmin((f-bin_edges)**2), len(bin_edges)-2)
                    prior = w*(Pfr[0][bidx])**spk
                    pr = 1. - pp
                    ns = pp/(pp + prior*pr)
                    newscore.append(ns)
                else:
                    newscore.append(pp)
            return np.array(newscore)

        probs = self.predict_proba(pfds)
        freqs = [1./getp0(p) for p in pfds]
        #print [p.extracted_feature.keys() for p in pfds]
        #freqs = [1./p.extracted_feature["ratings:['period']"] for p in pfds]

        newprobs = adjustscore(probs, freqs)

        return np.array([0. if res[1] == 0. else eval(self.score_mapper % res[1]) for res in newprobs])

        
    def score(self, pfds, target, F1=True):
        """
        return the mean of success array [1,0,0,1,...,1], where 1 is being right, and 0 is being wrong.
        """
        if not target.ndim == 1:
            target = target[...,0]#feature labeling
        predict = self.predict(pfds)
        if not F1:
            return np.mean(np.where(predict == target, 1, 0))
        else:
            P = np.mean(predict[target == 1])
            R = np.mean(target[predict == 1])
            F1score = 2 * P * R / (P + R)
            #print 'returnning F1:', F1
            #if F1 < 0.1:
                #print predict
                #print target
            return F1score



class classifier(object):
    """
    A class designed to be mixed in with the classifier class, to give it a feature property to specifiy what feature to extract.
    Usage:
    class svmclf(classifier, svm.SVC):
        orig_class = svm.SVC
        pass
    When initialize the classifier, remember to specify the feature like this:
    clf1 = svmclf(gamma=0.1, C=0.8, scale_C=False, feature={'phasebins':32})

    the feature has to be a diction like {'phasebins':32}, where 'phasebins' being the name of the feature, 32 is the size.
    """
    targetmap={'phasebins':1, 'DMbins':2, 'intervals':3, 'subbands':4, }
    def __init__(self, feature=None, use_pca=False, n_comp=12, **kwds):
        if feature == None:
            raise MyError(None)
        self.feature = feature
        self.use_pca = use_pca
        self.n_components = n_comp
        super(classifier, self).__init__( **kwds)

    def fit(self, pfds, target, randomshift=False):
        """
        args: pfds, target
        pfds: the training pfds
        target: the training targets
        randomshift: add a random shift to the phase, otherwise use the phase .5 aligned feature
        """
        MaxN = max([self.feature[k] for k in self.feature])
        feature = [k for k in self.feature if self.feature[k] == MaxN][0]
        #print '%s %s MaxN:%s'%(self.orig_class, self.feature, MaxN)
        #shift = random.randint(0, MaxN-1)
        shift = random.randint(0, MaxN-1, len(pfds))
        if not randomshift:
            shift *= 0
        Nspam = 3

        if feature in ['phasebins', 'timebins', 'freqbins']:
            #print '%s %s 1D shift:%s'%(self.orig_class, self.feature, shift)
            data = np.array([np.roll(pfd.getdata(**self.feature), shift[i])  for i, pfd in enumerate(pfds)])
        elif feature in ['intervals', 'subbands']:
            #print '%s %s 2D shift:%s'%(self.orig_class, self.feature, shift)
            if not randomshift:
                data = np.array([np.roll(pfd.getdata(**self.feature).reshape(MaxN, MaxN), shift[i], axis=1).ravel() for i, pfd in enumerate(pfds)])
            else:
                data = np.vstack([np.array([np.roll(pfd.getdata(**self.feature).reshape(MaxN, MaxN), shift, axis=1).ravel() for shift in random.randint(0, MaxN-1, Nspam)]) for i, pfd in enumerate(pfds)])
            #print data.shape
        else:
            data = np.array([pfd.getdata(**self.feature) for pfd in pfds])
        current_class = self.__class__
        self.__class__ = self.orig_class
        try:
            if target.ndim == 1:
                mytarget = target
            else:
                mytarget = target[...,classifier.targetmap[self.feature.keys()[0]]]
                
            if self.use_pca:
                self.pca = PCA(n_components=self.n_components).fit(data[mytarget == 1])
                data = self.pca.transform(data)

            if feature in ['intervals', 'subbands'] and randomshift:
                exptargets = np.array([ [t]*Nspam for t in mytarget]).ravel()
                mytarget = exptargets
            results = self.fit( data, mytarget)
        except KeyboardInterrupt as detail:
            import sys
            print sys.exc_info()[0], detail
        finally:
            self.__class__ = current_class

        return results
        #return self.orig_class.fit(self, data, target)

    def predict(self, pfds):
        """
        args: 
        pfds: list of pfddata objects

        Returns: array(Nsamples), giving the most-likely class
        """
        if not type(pfds) in [list, np.ndarray]:
            pfds = [pfds]
        data = np.array([pfd.getdata(**self.feature) for pfd in pfds])
        #self.test_data = data
        current_class = self.__class__
        self.__class__ = self.orig_class
        if self.use_pca:
            data = self.pca.transform(data)
        results =  self.predict(data)
        self.__class__ = current_class
        return results
        #return self.orig_class.predict(self, data)
        
    def predict_proba(self, pfds):
        """
        predict_proba(self, pfds) classifier method
        Compute the likehoods each possible outcomes of samples in T.
    
        The model need to have probability information computed at training
        time: fit with attribute `probability` set to True.
        
        Parameters
        ----------
        pfds: list of pfddata objects
        
        Returns
        -------
        X : array-like, shape = [n_samples, n_classes]
        Returns the probability of the sample for each class in
        the model, where classes are ordered by arithmetical
        order.
        
        Notes:
        ------
        * for NN, the probability isn't normalized across the classes because
              we are returning the activation of each neuron

        """ 
        if not type(pfds) in [list, np.ndarray]:
            pfds = [pfds]

        data = np.array([pfd.getdata(**self.feature) for pfd in pfds])
        current_class = self.__class__
        self.__class__ = self.orig_class
        if self.use_pca:
            data = self.pca.transform(data)
        results =  self.predict_proba(data)
        self.__class__ = current_class
        #AAR: compatible with multi-class (fixed)
        return results 

    def score(self, pfds, target, F1=True):
        """
        args: pfds, target
        pfds: the testing pfds
        target: the testing targets
        """
        #if 'pfds' in self.__dict__ and np.array(self.test_pfds == pfds).all() and str(self.feature) == self.last_feature:
            #print 'in score, skipping extract'
            #data = self.data
        #else:
            #print 'in score, not skipping extract'
            #data = np.array([pfd.getdata(**self.feature) for pfd in pfds])
            #self.test_pfds = tuple(pfds)
            #self.data = data
            #self.last_feature = str(self.feature)
        if not target.ndim == 1:
            target = target[...,0]#feature labeling
        data = np.array([pfd.getdata(**self.feature) for pfd in pfds])
        current_class = self.__class__
        self.__class__ = self.orig_class
        if self.use_pca:
            data = self.pca.transform(data)
        #results =  self.score(data, target)
        predict = self.predict(data)
        if not F1:
            F1score = np.mean(np.where(predict == target, 1, 0))
        else:
            P = np.mean(predict[target == 1])
            R = np.mean(target[predict == 1])
            F1score = 2 * P * R / (P + R)
            #print 'returnning F1:', F1
            #if F1 < 0.1:
                #print predict
                #print target
        self.__class__ = current_class
        return F1score
        #return super(classifier, self).score(data, target)
        #return self.orig_class.score(self, data, target)

class svmclf(classifier, svm.SVC):
    """
    the mix-in class for svm.SVC
    """
    orig_class = svm.SVC
    pass

class LRclf(classifier, linear_model.LogisticRegression):
    """
    the mix-in class for linear_model.LogisticRegression
    """
    orig_class = linear_model.LogisticRegression
    pass

class pnnclf(classifier, pnn.NeuralNetwork):
    """ 
    the mixed in class for pnn.NeuralNetwork
    """
    orig_class = pnn.NeuralNetwork
    pass

class dtreeclf(classifier, tree.DecisionTreeClassifier):
    """ 
    the mixed in class for DecisionTree
    """
    orig_class = tree.DecisionTreeClassifier
    pass

class ranforclf(classifier, ensemble.RandomForestClassifier):
    """ 
    the mixed in class for DecisionTree
    """
    orig_class = ensemble.RandomForestClassifier
    pass

class cnnclf(classifier, skcnn.MetaCNN):
    """
    the mixed in class for a convolutional neural network
    """
    orig_class = skcnn.MetaCNN
    pass

class adaboost(object):
    """
    a class to help with ensembles. 
    This class implements the adaboost method, determining the optimal weighting
    of the ensemble to maximize overall performance.

    Notes:
    Works for multi-class systems, but weights are only calculated/applied
    to class 1 objects.

    refer to http://en.wikipedia.org/wiki/Adaboost for more information
    
    Optionally:
    init with platt=True, for Platt initiationalizatoin (arxiv.org/pdf/1207.1403.pdf)
    though this hasn't helped for the PFD files

    """
    def __init__(self, platt=False):
        #use platt calibration to help get a predict_proba
        #arxiv.org/pdf/1207.1403.pdf
        if platt:
            self.platt = linear_model.LogisticRegression(penalty='l2')
        else:
            self.platt = None #doesn't work well in our case

    def fit(self, preds, targets):
        """
        use the adaboost to determine the optimal weights for
        the ensemble.

        We store the optimal weights in self.weights, later used in 
        self.predict

        Args:
        preds : [nsamples x npredictions]
        targets : [nsamples]

        Note:
        we accept labels (0,1), but process on (-1,1) labels

        """
        if self.platt != None:
        #split the data into training and x-val (for predict_proba fit)
            from random import shuffle
            L = len(targets)
            index = range(L)
            cut = int(.8*L)  #80pct training, 20pct x-val
            while 1:
                shuffle(index)
                train_idx = index[:cut]
                train_target = targets[train_idx]
                train_preds = preds[train_idx]

                test_idx = index[cut:]
                test_target = targets[test_idx]
                test_preds = preds[test_idx]
            
                if len(np.unique(train_target)) == len(np.unique(test_target)):
                    break
        else:
            # we don't need train/test split
            train_target = targets
            train_preds = preds
        
        if train_preds.ndim == 1:
            npreds = train_preds.shape[0]
        else:
            npreds = train_preds.shape[1]
        
        #'True' for wrong prediction, 'False' for correct prediction
        Wrong_pred = np.transpose([v != train_target for v in train_preds.transpose()]) 

        #remap predictions/targets from 0 to -1 if necessary
        y = np.where(train_target != 1, -1, 1)
        preds2 = np.where(train_preds != 1, -1,1)

        #indicator function  or scouting matrix(1 for wrong, 0 for right prediction)
        I = np.where(Wrong_pred, 1., 0.)

        clfs = {}
        alphas = {}
        #Weight of each data point
        D = np.ones(len(y), dtype=np.float)/len(y)
        allclfs = set(range(npreds))
        for t in range(npreds):
            # find best remaining classifier
            idcs = list(allclfs - set(clfs.values()))
            W_e = np.dot(D,I) 
            best = np.argmax(np.abs(0.5-W_e[idcs])) #same as np.argmin(W_e[idcs])
            h_t = np.where(W_e == W_e[idcs][best])[0][0]

            e_t = W_e[h_t] 
#            print h_t, e_t, W_e
            if np.abs(0.5 - e_t) <= .10: break # we've done enough, error<10%ish
                                                # lowering threshold brings in more error

            clfs[t] = h_t
            alpha_t = np.log((1.-e_t)/e_t)/2.
            alphas[t] = alpha_t
            
            Z_t = D*np.exp(-alpha_t*y*preds2[:,h_t]).sum()
            D = D*np.exp(-alpha_t*y*preds2[:,h_t])/Z_t

        #append the classifier weights (in order of list_of_AIs)
        if len(clfs) <= 2:
            #if everything was poor, give equal weighting
            w = np.ones(npreds, dtype=float)/npreds 
        else:
            w = np.zeros(npreds, dtype=float)
        for k, v in clfs.iteritems():
            w[v] = alphas[k]
        self.weights = w
        self.clfs = clfs
        self.alphas = alphas

        #finally, fit the platt calibration for predict_proba functionality
        if self.platt != None:
            test_preds2 = np.where(test_preds != 1, -1,1)
            this_preds = np.transpose([np.where(np.dot(test_preds2, self.weights) > 0, 1, 0)]) 
            self.platt.fit( this_preds, test_target)

    def predict(self, list_of_predictions):
        """
        apply the adaboost weights and form the final hypothesis
        H(x) = sign( \sum_classifier weight(i) * h_i(x) )
        
        Note:
        although we accept labels of (0,1) and (-1,1)
        we only return labels (0, 1)
        """
        #GBC assumes labels are -1, +1, so re-map
        if 0 in np.unique(list_of_predictions):
            tmp = np.where(list_of_predictions != 1, -1, 1)
        else:
            tmp = list_of_predictions
        return  np.where(np.dot(tmp, self.weights) >= 0., 1, 0)

    def predict_proba(self, lops):
        """
        following arxiv.org/pdf/1207.1403.pdf
        
        *use a Platt calibration (done in 'fit') to provide
        a predict_proba (actually, this didn't work well)
        *apply the adaboost.weights to the predict_proba class 1
        uniform weight to all other classes (which we largely ignore anyways)

        Args:
        lops: the predict_proba's from the list_of_AIs
        
        Returns:
        array of [nsamples x nclasses]

        Notes:
        The final classifier operates as H(x) = sign(\sum_i w[i] h_i(x))
            where 'i' is over classifiers.
        Here we simply return (\sum_i w[i] h_i(x)), **so this isn't really a prob. distribution**
        negative probs. are possible (eg. if everyone was a "perfect liar")
        
        """
        if isinstance(lops, list):
            lops = np.array(lops)

        npreds = len(self.weights)
        if lops.ndim == 2:
            nclass = lops.shape[1] // npreds
            nsamples = lops.shape[0]
        else:
            nclass = lops.shape[0] // npreds
            nsamples = 1
            lops = np.array([lops])

       
        if self.platt is not None:
        #this techniques doesn't do that well
            f = np.transpose([self.predict(lops)]) #[nsamples x 1] 
            return  self.platt.predict_proba(f)
        else:
            #self.weight is for 1 class, lops may have several
            npreds = len(self.weights)
            if lops.ndim == 2:
                nclass = lops.shape[1] // npreds
                nsamples = lops.shape[0]
            else:
                nclass = lops.shape[0] // npreds
                nsamples = 1

            # H(x) works on sign(sum_i w[i]h_i(x))
            # so shift all predictions (0 < lops < 1) to (-1 < lops < 1)
            lops = 2.*lops - 1.

            #so repeat the weights nclass times
            #weights are only for 'class 1', so use uniform weight on non-'1' classes
            w = np.ones((npreds,nclass), dtype=np.float)/float(npreds)
            w[:,1] = self.weights
            f = np.transpose([np.dot( lops[:,c::nclass], v)\
                                  for c, v in enumerate(w.transpose())])
            #use sigmoid to get final predict_proba
            return 1./(1.0 + np.exp(-f))

def extractfeatures(AIlist, pfds):
    """
    given a list of AIs (eg. combinedAI.list_of_AIs)
    and a list of pfds (class pfdreader),
    pre-extract all the useful features.
    This is meant to reduce disk i/o and calls to pfd.dedisperse()
    #Auto extract p0 #2013/04/29
    """

    #determine features to extract from pfd
    features = {}
    vargf = [{'ratings':['period']}] # auto extract P0
    items = []
    for clf in AIlist:
        items.extend(clf.feature.items())

    newf = set([ '%s:%s'% (f,v)  for f,v in items]) - set(pfds[0].extracted_feature.keys())
    for p in newf:
        f,v = p.split(':')
        vargf.append({f:int(v)})
    if len(vargf) > 0:
        def getfeature(pfd):
            pfd.getdata(*vargf, **features)
            return pfd
        resultdict = threadit(getfeature, [[p] for p in pfds])
        for n, pfd in resultdict.iteritems():
            if pfd == None:
                print 'ZeroDivisionError: ', pfds[n].pfdfile
                raise ZeroDivisionError
            pfds[n] = pfd

def threadpredict(AIlist, pfds):
    """
    Args:
    AIlist : list of trained classifiers
    pfds : list of pfds
    out : output format, one of 'transpose' or 'hstack'
    """
    def predictfunc(pfds, clf):
        return clf.predict(pfds)
    resultdict = threadit(predictfunc, [[pfds, clf] for clf in AIlist])
    return np.transpose([resultdict[n] for n in range(len(AIlist))])
        
def threadpredict_proba(AIlist, pfds):
    """
    Args:
    AIlist : list of trained classifiers
    pfds : list of pfds
    """
    def predict_prob(clf):
        #try:
        p = clf.predict_proba(pfds)
        #except:
            #print 'Alarm!!!'
        return p
    resultdict = threadit(predict_prob, [[clf] for clf in AIlist])
    return np.hstack([resultdict[n] for n in range(len(AIlist))])


class MyError(Exception):
    def __init__(self, note):
        self.note = note
    def __str__(self):
        if self.note is None:
            return repr("must specify the feature used by this classifier")
        else:
            return repr(self.note)
        
