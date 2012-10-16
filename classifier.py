import random
import numpy as np
from sklearn.decomposition import RandomizedPCA as PCA
from sklearn import svm, linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from ubc_AI import pulsar_nnetwork as pnn 

class combinedAI(object):
    """
    A class to combine different AIs, and have them operate as one
    """
    def __init__(self, list_of_AIs, strategy='vote', nvote=None, **kwds):
        """
        inputs
        list_of_AIs: list of classifiers
        strategy: one of ['union', 'vote', 'l2', 'svm', 'forest', 'tree', 'nn']
        
        Notes:   
        *'l2' uses LogisticRegression on the prediction matrix from list_of_AIs,
             and makes final prediction from the l2(predictions)
        *'svm' uses SVM on the prediction matrix from the list_of_AIs,
             and makes the final prediciton from SVM(predictions)
        *'forest' uses sklearn.ensemble.RandomForestClassifier
        *'tree' DecisionTreeClassifier
        *'nn' uses a 1-layer, N/2-neuron classifier [N=len(list_of_AIs)]
        *if strategy='vote' and nvote=None, 
           determine best nvote value during self.fit (but this doesn't work good)

        """
        self.AIonAIs = ['l2','svm','forest','tree','nn']
        self.list_of_AIs = list_of_AIs
        self.strategy = strategy
        if strategy == 'l2':
            self.AIonAI = linear_model.LogisticRegression(penalty='l2', **kwds)
        elif strategy == 'svm':
            self.AIonAI = svm.SVC(probability=True, **kwds)
        elif strategy == 'forest':
            self.AIonAI = RandomForestClassifier()
        elif strategy == 'tree':
            self.AIonAI = DecisionTreeClassifier()
        elif strategy == 'nn':
            n = max(1,int(len(list_of_AIs)/2))
            self.AIonAI = pnn.NeuralNetwork(design=[n,2], **kwds)
                    
        self.nvote = nvote

    def fit(self, pfds, target, **kwds):
        """
        args: [list of pfd instances], target
        """
        if target.ndim == 1:
            psrtarget = target
        else:
            psrtarget = target[...,0]
        train_preds = []
        for clf in self.list_of_AIs:
            clf.fit(pfds,target, **kwds)

        if (self.strategy in self.AIonAIs):
            if (self.strategy not in ['tree', 'forest']):
                #use predict_prob
                predictions = [clf.predict_proba(pfds)\
                                   for clf in self.list_of_AIs] #npred x nsamples
            else:
                #use predict
                predictions = [clf.predict(pfds)\
                                   for clf in self.list_of_AIs]

            predictions = np.array(predictions).transpose() #nsamples x npred
            self.AIonAI.fit(predictions, psrtarget)
            
# choose 'nvote' that maximizes the trianing-set performance                
        if self.strategy == 'vote' and self.nvote == None:
            train_preds = np.array(train_preds).transpose() #nsamples x nclassifiers
            score = 0.
            for i in range(len(self.list_of_AIs)):
                pct = (i+1.)/len(self.list_of_AIs)
                avepred = np.where(train_preds.sum(axis=1) > pct, 1, 0)
                this_score = np.mean(np.where(avepred == psrtarget, 1, 0))
                if this_score > score:
                    self.nvote = i + 1
                    score = this_score

                
    def predict(self, test_pfds, pred_mat=False ):
        """
        args: [list of test pfd instances], test target
        optionally: pred_mat = True returns the [nsamples x npredictions] matrix
                               so you can run your own prediction combo schemes
                               (default False)
        """
        if not type(test_pfds) in [list, np.ndarray]:
            print "warniing: changing test_pfds from type %s to list" % (type(test_pfds))
            test_pfds = [test_pfds]
        
        if (self.strategy in self.AIonAIs) and (self.strategy not in ['tree', 'forest']):
            #use predict_proba for our non-tree/forest AI_on_AI classifier, 
            list_of_predicts = [clf.predict_proba(test_pfds) for clf in self.list_of_AIs]
        else:
            list_of_predicts = [clf.predict(test_pfds) for clf in self.list_of_AIs]
        self.list_of_predicts = np.array(list_of_predicts).transpose() #nsamp x npred

        self.predictions = []
        if self.strategy == 'union':
            #return '1'=pulsar if any AI voted yes, otherwise '0'=rfi
            self.predictions = np.where((lop==1).sum(axis=1)>0, 1, 0)

        elif self.strategy == 'vote':
            predict = self.list_of_predicts.mean(axis=1)#[nsamples x npred]
            npreds = float(len(self.list_of_AIs))
            self.predictions = np.where( predict > self.nvote/npreds, 1, 0)
            
        elif self.strategy in self.AIonAIs:
            predict = self.list_of_predicts #[nsamples x npred]
            self.predictions = self.AIonAI.predict(predict)

        #return np.array(self.predictions)
        if pred_mat:
            return self.list_of_predicts #[nsamples x npredictions]
        else:
            return self.predictions

    def predict_proba(self, pfds):
        """
        predict_proba(self, pfds) classifier method
        Compute the likehoods each possible outcomes of samples in T.
    
        The model need to have probability information computed at training
        time: fit with attribute `probability` set to True.
    
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        
        Returns
        -------
        X : array-like, shape = [n_samples, n_classes]
        Returns the probability of the sample for each class in
        the model, where classes are ordered by arithmetical
        order.
        
        Notes
        -----
        """
        
        if self.strategy not in self.AIonAIs:
            result = np.sum(np.array([clf.predict_proba(pfds) for clf in self.list_of_AIs]), axis=0)/len(self.list_of_AIs)
        else:
            predicts = [clf.predict(pfds) for clf in self.list_of_AIs]
            predicts = np.array(predicts).transpose()
            #AAR: not compatible with multi-class (future fix)
            result = self.AIonAI.predict_proba(predicts)[...,1]
        return result
        
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
    def __init__(self, feature=None, use_pca=False, n_comp=12, *args, **kwds):
        if feature == None:
            raise "must specifiy the feature used by this classifier!"
        self.feature = feature
        self.use_pca = use_pca
        self.n_components = n_comp
        self.targetmap = {'phasebins':1, 'DMbins':2, 'intervals':3, 'subbands':4, }
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
        if target.ndim == 1:
            results = self.fit( data, target)
        else:
            #print 'feature label recongnized!', self.targetmap[self.feature.keys()[0]], target.shape
            results = self.fit( data, target[...,self.targetmap[self.feature.keys()[0]]])
        self.__class__ = current_class
        return results
        #return self.orig_class.fit(self, data, target)

    def predict(self, pfds):
        """
        args: pfds
        pfds: the testing pfds
        """
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
    X : array-like, shape = [n_samples, n_features]
    
    Returns
    -------
    X : array-like, shape = [n_samples, n_classes]
        Returns the probability of the sample for each class in
        the model, where classes are ordered by arithmetical
        order.
    
    Notes
    -----
        """
        data = np.array([pfd.getdata(**self.feature) for pfd in pfds])
        current_class = self.__class__
        self.__class__ = self.orig_class
        if self.use_pca:
            data = self.pca.transform(data)
        results =  self.predict_proba(data)
        self.__class__ = current_class
        #AAR: not compatible with multi-class (future fix)
        return results[...,1]

    def score(self, pfds, target, F1=True):
        """
        args: pfds, target
        pfds: the testing pfds
        target: the testing targets
        """
        if not target.ndim == 1:
            #print 'feature label recongnized!'
            target = target[..., self.targetmap[self.feature.keys()[0]]]
        #if 'test_pfds' in self.__dict__ and np.array(self.test_pfds == pfds).all() and str(self.feature) == self.last_feature:
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
    the mix-in class for svm.SVC
    """
    orig_class = linear_model.LogisticRegression
    pass

class pnnclf(classifier, pnn.NeuralNetwork):
    """ 
    the mixed in class for pnn.NeuralNetwork
    """
    orig_class = pnn.NeuralNetwork
    pass

class dtreeclf(classifier, DecisionTreeClassifier):
    """ 
    the mixed in class for DecisionTree
    """
    orig_class = DecisionTreeClassifier
    pass
