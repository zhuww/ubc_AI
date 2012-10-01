import random
import numpy as np
from sklearn.decomposition import RandomizedPCA as PCA
from sklearn import svm, linear_model, tree, ensemble
from sklearn.ensemble import GradientBoostingClassifier as GBC

from ubc_AI.training import split_data
from ubc_AI import pulsar_nnetwork as pnn 

class combinedAI(object):
    """
    A class to combine different AIs, and have them operate as one
    """
    def __init__(self, list_of_AIs, strategy='vote', nvote=None, **kwds):
        """
        inputs
        list_of_AIs: list of classifiers
        strategy: What to do with the prediction matrix from the list_of_AIs.
                One of ['vote', 'lr', 'svm', 'forest', 'tree', 'nn', 'adaboost', 'gbc', 'kitchensink']
                Default = 'vote'
        
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

        *if strategy='vote' and nvote=None, 
           determine best nvote value during self.fit (but this doesn't work good)

        """
        #things that require a 'fit'
        self.AIonAIs = ['lr', 'svm', 'forest', 'tree', 'nn', 'adaboost', 'gbc', 'kitchensink']
        #things that train on 'predict' instead of 'predict_proba'
        self.req_predict = ['adaboost', 'gbc']

        self.list_of_AIs = list_of_AIs
        self.strategy = strategy
        if strategy != 'vote' and strategy not in self.AIonAIs:
            raise "strategy %s is not recognized" % strategy
        if strategy == 'lr':
            #grid-search optimized
            self.AIonAI = linear_model.LogisticRegression(C=0.5, penalty='l1', **kwds)
        elif strategy == 'svm':
            #grid-search optimized
            self.AIonAI = svm.SVC(C=15, kernel='poly', degree=5, probability=True, **kwds)
        elif strategy == 'forest':
            self.AIonAI = ensemble.RandomForestClassifier(**kwds)
        elif strategy == 'tree':
            nleafs = len(list_of_AIs)/2
            self.AIonAI = tree.DecisionTreeClassifier(min_samples_leaf=nleafs,**kwds)
        elif strategy == 'nn':
            #grid-search optimized (2class)
            self.AIonAI = pnn.NeuralNetwork(gamma=15, design=[64], **kwds) 
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

        self.nvote = nvote
        self.nclasses = None #keep track of number of classes (determined in 'fit')

    def fit(self, pfds, target, **kwds):
        """
        args: [list of pfd instances], target

        Notes:
        following advice from http://en.wikipedia.org/wiki/Ensemble_learning
        we train each classifier on a subset of the training data
        
        
        """
        #train the individual classifiers on a random subset of data
        for clf in self.list_of_AIs:
            tr_data, tr_target, te_data, te_target = split_data(pfds, target, pct=0.75)
#            clf.fit(pfds, target, **kwds)
            clf.fit(tr_data, tr_target, **kwds)

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
                predictions = np.hstack([clf.predict_proba(pfds)\
                                             for clf in self.list_of_AIs]) #nsamples x (npred x nclasses)
            else:
                #use predict
                predictions = np.transpose([clf.predict(pfds)\
                                      for clf in self.list_of_AIs]) #nsamples x npred
            self.AIonAI.fit(predictions, target)

            
# choose 'nvote' that maximizes the trianing-set performance                
        if self.strategy == 'vote' and self.nvote == None:
            train_preds = []
            train_preds = np.array(train_preds).transpose() #nsamples x nclassifiers
            score = 0.
            for i in range(len(self.list_of_AIs)):
                pct = (i+1.)/len(self.list_of_AIs)
                avepred = np.where(train_preds.sum(axis=1) > pct, 1, 0)
                this_score = np.mean(np.where(avepred == target, 1, 0))
                if this_score > score:
                    self.nvote = i + 1
                    score = this_score


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
        
        if (self.strategy in self.AIonAIs) and self.strategy not in self.req_predict:
            #use predict_proba for AI_on_AI classifier, 
            list_of_predicts = np.hstack([clf.predict_proba(pfds)\
                                              for clf in self.list_of_AIs])#nsamples x (npred x classes)
        else:
            list_of_predicts = np.transpose([clf.predict(pfds)\
                                                 for clf in self.list_of_AIs]) #nsamples x npred

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

        if self.strategy not in self.AIonAIs:
            result = np.array([clf.predict_proba(pfds)\
                                 for clf in self.list_of_AIs]) #npreds x nsamples x nclasses
            result = result.mean(axis=0) #nsamples x nclasses
            
        else:
            #note: adaboost.predict_proba now accepts predict_proba inputs
            if self.strategy in self.req_predict and self.strategy != 'adaboost':
                predicts = np.transpose([clf.predict(pfds)\
                                             for clf in self.list_of_AIs]) #nsamples x nclasses
            else:
                predicts = np.hstack([clf.predict_proba(pfds)\
                                          for clf in self.list_of_AIs]) #nsamples x (npreds x nclasses)

            result = self.AIonAI.predict_proba(predicts) #nsamples x nclasses
        return result
        
    def score(self, pfds, target, F1=True):
        """
        return the mean of success array [1,0,0,1,...,1], where 1 is being right, and 0 is being wrong.
        """
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
    def __init__(self, feature=None, use_pca=False, n_comp=12, *args, **kwds):
        if feature == None:
            raise "must specifiy the feature used by this classifier!"
        self.feature = feature
        self.use_pca = use_pca
        self.n_components = n_comp
        super(classifier, self).__init__( *args, **kwds)

    def fit(self, pfds, target):
        """
        args: pfds, target
        pfds: the training pfds
        target: the training targets
        """
        MaxN = max([self.feature[k] for k in self.feature])
        feature = [k for k in self.feature if self.feature[k] == MaxN][0]
        #print '%s %s MaxN:%s'%(self.orig_class, self.feature, MaxN)
        shift = random.randint(0, MaxN-1)

        if feature in ['phasebins', 'timebins', 'freqbins']:
            #print '%s %s 1D shift:%s'%(self.orig_class, self.feature, shift)
            data = np.array([np.roll(pfd.getdata(**self.feature), shift) for pfd in pfds])
        elif feature in ['intervals', 'subbands']:
            #print '%s %s 2D shift:%s'%(self.orig_class, self.feature, shift)
            data = np.array([np.roll(pfd.getdata(**self.feature).reshape(MaxN, MaxN), shift, axis=1).ravel() for pfd in pfds])
        else:
            data = np.array([pfd.getdata(**self.feature) for pfd in pfds])
        current_class = self.__class__
        self.__class__ = self.orig_class
        if self.use_pca:
            self.pca = PCA(n_components=self.n_components).fit(data)
            data = self.pca.transform(data)
        results = self.fit( data, target)
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


class adaboost(object):
    """
    a class to help with ensembles. 
    This class implements the adaboost method, determining the optimal weighting
    of the ensemble to maximize overall performance.

    Notes:
    It only works for 2-class systems, and expects labels of (0,1)

    refer to http://en.wikipedia.org/wiki/Adaboost for more information
   
    """
    def __init__(self):
        #use platt calibration to help get a predict_proba
        #arxiv.org/pdf/1207.1403.pdf
        self.platt = linear_model.LogisticRegression(penalty='l2')

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
        D = np.ones(len(y))/len(y)
        allclfs = set(range(npreds))
        for t in range(npreds):
            # find best remaining classifier
            idcs = list(allclfs - set(clfs.values()))
            W_e = np.dot(D,I) 
            best = np.argmax(np.abs(0.5-W_e[idcs])) #same as np.argmin(W_e[idcs])
            h_t = np.where(W_e[idcs][best] == W_e)[0][0]

            e_t = W_e[h_t]/W_e.sum()
            if np.abs(0.5 - e_t) <= .425: break # we've done enough, error<17%ish
                                                # lowering threshold brings in more error

            clfs[t] = h_t
            alpha_t = np.log((1.-e_t)/e_t)/2.
            alphas[t] = alpha_t
            
            Z_t = D*np.exp(-alpha_t*y*preds2[:,h_t]).sum()
            D = D*np.exp(-alpha_t*y*preds2[:,h_t])/Z_t

        #append the classifier weights (in order of list_of_AIs)
        if len(clfs) == 0:
            #if everything was poor, give equal weighting
            w = np.ones(npreds, dtype=float)/npreds 
        else:
            #we give everyone a vote, just really small sometimes (overwritten below)
            w = np.ones(npreds, dtype=float)/sum(clfs.values())/npreds
        for k, v in clfs.iteritems():
            w[v] = alphas[k]
        self.weights = w
        self.clfs = clfs
        self.alphas = alphas

        #finally, fit the platt calibration for predict_proba functionality
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

        return  np.where(np.dot(tmp, self.weights) > 0, 1, 0)

    def predict_proba(self, lops):
        """
        following arxiv.org/pdf/1207.1403.pdf
        
        *use a Platt calibration (done in 'fit') to provide
        a predict_proba (actually, this didn't work well)
        *apply the adaboost.weights to the predict_proba class 1
        uniform weight to all other classes (which we largely ignore anyways)

        Args:
        lops: the yes/no (1,0) or (1,-1) list of predictions
        
        Returns:
        array of [nsamples x nclasses]
        """
        if isinstance(lops, list):
            lops = np.array(lops)
        if 0:
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
                
            #so repeat the weights nclass times
            #weights are only for 'class 1', so use uniform weight on non-'1' classes
            w = np.ones((npreds,nclass))
            w[:,1] = self.weights

            f = np.transpose([np.dot( lops[:,c::nclass], v)/v.sum()\
                                  for c, v in enumerate(w.transpose())])
            return f
