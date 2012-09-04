import random
import numpy as np
from sklearn.decomposition import RandomizedPCA as PCA
from sklearn import svm, linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy import mgrid

from ubc_AI import pulsar_nnetwork as pnn 

class combinedAI(object):
    """
    A class to combine different AIs, and have them operate as one
    """
    def __init__(self, list_of_AIs, strategy='vote', nvote=None, shift_predict=False):
        """
        inputs
        list_of_AIs: list of classifiers
        strategy: one of ['union', 'vote', 'l2', 'svm', 'forest', 'tree', 'nn']
        shift_predict: shift each feature along phasebin, getting predict_proba
                        with each shift. Return 'predict/predict_proba' for best
                        phase (if the different features maximize at same location).
                        *All classifiers (expect DM-related) in list_of_AIs need this option)
        
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
        self.shift_predict = shift_predict
        if strategy == 'l2':
            self.AIonAI = linear_model.LogisticRegression(penalty='l2')
        elif strategy == 'svm':
            self.AIonAI = svm.SVC(probability=True)
        elif strategy == 'forest':
            self.AIonAI = RandomForestClassifier()
        elif strategy == 'tree':
            self.AIonAI = DecisionTreeClassifier()
        elif strategy == 'nn':
            n = max(1,int(len(list_of_AIs)/2))
            self.AIonAI = pnn.NeuralNetwork(gamma=1./n,design=[n,2])
                    
        self.nvote = nvote

    def fit(self, pfds, target, **kwds):
        """
        args: [list of pfd instances], target
        """
        train_preds = []
        for clf in self.list_of_AIs:
            clf.fit(pfds,target, **kwds)

        if (self.strategy in self.AIonAIs):
            if (self.strategy not in ['tree', 'forest']):
                #use predict_prob
                predictions = [clf.predict_proba(pfds, shift_predict=False)\
                                   for clf in self.list_of_AIs] #npred x nsamples
            else:
                #use predict
                predictions = [clf.predict(pfds, shift_predict=False)\
                                   for clf in self.list_of_AIs]

            predictions = np.array(predictions).transpose() #nsamples x npred
            self.AIonAI.fit(predictions, target)
            
# choose 'nvote' that maximizes the trianing-set performance  
# Note: this should be avoided...               
        if self.strategy == 'vote' and self.nvote == None:
            train_preds = np.array(train_preds).transpose() #nsamples x nclassifiers
            score = 0.
            for i in range(len(self.list_of_AIs)):
                pct = (i+1.)/len(self.list_of_AIs)
                avepred = np.where(train_preds.sum(axis=1) > pct, 1, 0)
                this_score = np.mean(np.where(avepred == target, 1, 0))
                if this_score > score:
                    self.nvote = i + 1
                    score = this_score

                
    def predict(self, test_pfds, pred_mat=False, shift_predict=None):
        """
        args: [list of test pfd instances], test target
        optionally: pred_mat = True returns the [nsamples x npredictions] matrix
                               so you can run your own prediction combo schemes
                               (default False)

        Notes:
        shift_predict is provided here because we don't us 
        self.shift_predict in the fit routine and need to override it

        """
        if shift_predict is None:
            shift_predict = self.shift_predict

        if not type(test_pfds) in [list, np.ndarray]:
            print "warning: changing test_pfds from type %s to list" % (type(test_pfds))
            test_pfds = [test_pfds]
        
        if not shift_predict:
            #then we do regular voting, taking PFD data as-is (no phase-shifting)

            if (self.strategy in self.AIonAIs) and (self.strategy not in ['tree', 'forest']):
            #use predict_proba for our non-tree/forest AI_on_AI classifier, 
                list_of_predicts = [clf.predict_proba(test_pfds, shift_predict=False)\
                                        for clf in self.list_of_AIs]
            else:
                list_of_predicts = [clf.predict(test_pfds, shift_predict=False)\
                                        for clf in self.list_of_AIs]
        else:
            list_of_predicts = self.shift_predictions(test_pfds, shift_predict)

        #get [nsamples x npred]
        self.list_of_predicts = np.array(list_of_predicts).transpose() 
        self.predictions = self.vote(self.list_of_predicts)

        if pred_mat:
            return self.list_of_predicts #[nsamples x npredictions]
        else:
            return self.predictions

    def shift_predictions(self, test_pfds, shift_predict=True, retmat=False):
        """
        the heart of the "predict" and "predict_proba" routines
        if we are doing "shift_predict"
        Args:
        test_pfds : list of pfddata files
        shift_predict : do shift predict or not (default True)
        retmat : return a matrix of [nsamples x npredictions x max(nbins)] 
                 if a classifier is non-phase related (eg. DMbins), it is included
                 but its result is smply repeated max(nbins) times.
                 This is useful to do your own analysis on the Prob(phase shift)

        Returns:
        the optimal probability after shifting all classifiers
        to each phase bin.
        This is an array of [nsamples x npredictions(best_phase)]

        """
        #then list_of_predicts is [nclassifiers][nbins x nsamples]
        if (self.strategy in self.AIonAIs) and (self.strategy not in ['tree', 'forest']):
        #use predict_proba for our non-tree/forest AI_on_AI classifier, 
            list_of_predicts = [clf.predict_proba(test_pfds, shift_predict=shift_predict)\
                                    for clf in self.list_of_AIs\
                                    if clf.feature.keys()[0] != 'DMbins']
            dtype = np.float
        else:
            list_of_predicts = [clf.predict(test_pfds, shift_predict=shift_predict)\
                                     for clf in self.list_of_AIs\
                                    if clf.feature.keys()[0] != 'DMbins']
            dtype = np.int
        lop = np.array(list_of_predicts)

    #sample all Probs vs phase to the largest grid
        nbins = [len(i) for i in lop]
        max_nbin = max(nbins)
        coords = mgrid[0:1:1j*max_nbin]
        nclf = len(lop) #number of non-DM classifiers
        nsamples = lop[0][0].size
#change order of indices to [nsamples, nclassifers, nbins]
        lop_dwn = np.zeros((nsamples, nclf, max_nbin), dtype=dtype)
        for clfi, dat in enumerate(lop):
            m = len(dat)
            x = mgrid[0:1:1j*m]
            data = np.array(dat).transpose() #[data] = [nsamples x max_nbin]
            for si, sv in enumerate(data):
                lop_dwn[si,clfi,:] = np.interp(coords, x, sv)
        #add up all the predictions and find first index with best prob
        lop_dwn_bestbin = lop_dwn.sum(axis=1).argmax(axis=1) #gives best bin for [nsamples] on min_nbin grid

     #now go back to our predictions, filling in with the phase-shifted votes:
        list_of_predicts = []
        if retmat:
            retmat_lop = np.zeros((nsamples, len(self.list_of_AIs), max_nbin))
        n_nondm = 0
        for clfi, clf in enumerate(self.list_of_AIs):
            if clf.feature.keys()[0] == 'DMbins':
                if (self.strategy in self.AIonAIs) and (self.strategy not in ['tree', 'forest']):
                    list_of_predicts.append( clf.predict_proba(test_pfds) )
                    if retmat:
                        retmat_lop[:, clfi, :] = np.array([clf.predict_proba(test_pfds)\
                                                               for n in range(max_nbin)]).transpose()
                else:
                    list_of_predicts.append( clf.predict(test_pfds) )
                    if retmat:
                        retmat_lop[:, clfi, :] = np.array([clf.predict(test_pfds)\
                                                               for n in range(max_nbin)]).transpose()
                    
            else:
                # use Prob(bestphase) calculated previously
                samps = []
                for bi, bv in enumerate(lop_dwn_bestbin):
                    if retmat:
                        dat = lop[n_nondm]
                        m = len(dat)
                        x = mgrid[0:1:1j*m]
                        data = np.array(dat).transpose() #[nsamples x max_nbin]
                        for si, sv in enumerate(data):
                            retmat_lop[si, clfi, :] = np.interp(coords, x, sv)
                    if 1:
                        #use original predictions
                        orig_bin = int(float(bv)/max_nbin*nbins[n_nondm]) 
                        samps.append(lop[n_nondm][orig_bin][bi]) 
                    else:
                        #use downsampled bin (bv)
                        samps.append(lop_dwn[bi,n_nondm,bv])   
                n_nondm += 1
                list_of_predicts.append( samps ) #[npred x nsamples
        if retmat:
            return retmat_lop #[nsamples x len(list_of_AIs) x max_nbins]
        else:
            return np.array(list_of_predicts) #[npred x nsamples]


    def vote(self, pred_mat):
        """
        given the prediction matrix pred_mat = [nsamples x npred]
        perform the voting strategy, returning the final prediction

        """
        predictions = []
        if self.strategy == 'union':
        #return '1'=pulsar if any AI voted yes, otherwise '0'=rfi
            predictions = np.where((pred_mat==1).sum(axis=1)>0, 1, 0) #nsamples
        elif self.strategy == 'vote':
            predict = pred_mat.sum(axis=1)#[nsamples x npred]
            npreds = float(len(self.list_of_AIs))
            predictions = np.where( predict > self.nvote/npreds, 1, 0)

        elif self.strategy in self.AIonAIs:
            predict = pred_mat #[nsamples x npred]
            predictions = self.AIonAI.predict(predict)
        return predictions

    def predict_proba(self, pfds, shift_predict=None):
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
        shift_predict is provided here because we don't us 
        self.shift_predict in the fit routine and need to override it
        """
        if shift_predict is None:
            shift_predict = self.shift_predict

        if not shift_predict:
            if self.strategy not in self.AIonAIs:
                result = np.sum(np.array([clf.predict_proba(pfds)\
                                              for clf in self.list_of_AIs]), axis=0)/len(self.list_of_AIs)
            else:
                predicts = [clf.predict(pfds)\
                                for clf in self.list_of_AIs]
                predicts = np.array(predicts).transpose()
            #AAR: not compatible with multi-class (future fix)
                result = self.AIonAI.predict_proba(predicts)[...,1]
        else:
            list_of_predicts = self.shift_predictions(test_pfds, shift_predict)
            result = self.AIonAI.predict_proba(list_of_predicts)[...,1]
            
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

    def plot_shiftpredict(self, pfd):
        """
        Accepts: a combinedAI object and a single pfd file.
        Plots the probability distribution
        for the each classifier as we shift in phase

        """
        import pylab as plt
        from itertools import cycle
        lines = ["-",":","-.","--"]
        linecycler = cycle(lines)
        if not isinstance(pfd, type(list())):
            pfd = [pfd]

        lop = [clf.predict_proba(pfd, shift_predict=True)\
                                for clf in self.list_of_AIs\
                                if clf.feature.keys()[0] != 'DMbins']
        n_nondm = 0
        fig = plt.figure()
        ax = fig.add_subplot(221)
        for clfi, clf in enumerate(self.list_of_AIs):
            if clf.feature.keys()[0] != 'DMbins':
                v = lop[n_nondm]
                x = mgrid[0:1:len(v)*1j]
                lbl = str(type(clf)).split('.')[-1].strip('>').strip('\'')
                lbl += ' %s' % clf.feature
                ax.plot(x,v, next(linecycler),label=lbl)
                n_nondm += 1
        ax.set_xlabel('phase')
        ax.set_ylabel('Probability')
        ax.set_title('non-gridded prob distr')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        ax = fig.add_subplot(223)
        nbins = [len(i) for i in lop]
        min_nbin = min(nbins)
        coords = mgrid[0:1:1j*min_nbin]
        n_nondm = 0
        linecycler = cycle(lines)
        for clfi, clf in enumerate(self.list_of_AIs):
            if clf.feature.keys()[0] != 'DMbins':
                v = lop[n_nondm]
                m = len(v)
                x = mgrid[0:1:len(v)*1j]
                data = np.interp(coords, x, np.array(v).transpose()[0])
                lbl = str(type(clf)).split('.')[-1].strip('>').strip('\'')
                lbl += ' %s' % clf.feature
                ax.plot(coords, data, next(linecycler) ,label=lbl)
                n_nondm += 1
        ax.set_xlabel('phase')
        ax.set_ylabel('Probability')
        ax.set_title('gridded prob distr')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

        #now try doing a surface plot (z = f(phase, classifier))
        plt.clf()
        fig = plt.figure()
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.gca(projection='3d')#fig.add_subplot(111, projection='3d')
        x = mgrid[0:1:100j]#coords #phase
        y = range(len(lop))
        X, Y = np.meshgrid(x,y)
        def fun(x, y):
            v = lop[int(y)]
            p = mgrid[0:1:len(v)*1j]
            data = np.interp(x, p, np.array(v).transpose()[0])
            return data
        zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
        Z = zs.reshape(X.shape)
        ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
        cset = ax.contour(X, Y, Z, zdir='z', offset=0)
        cset = ax.contour(X, Y, Z, zdir='x', offset=0)
        cset = ax.contour(X, Y, Z, zdir='y', offset=len(lop)-1)
        ax.set_xlabel('Phase')
        ax.set_ylabel('Classifier')
        ax.set_zlabel('Probability')
        plt.show()

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
    
    Notes:
    predict and predict_proba accept a 'shift_predict' argument,
    returning prediction matrix (nbins, nsamples). This matrix is 
    used for x-comparison in combinedAI to find best phase.
   
    """
    def __init__(self, feature=None, use_pca=False, n_comp=12, *args, **kwds):
        if feature == None:
            raise "must specifiy the feature used by this classifier!"
        self.feature = feature
        self.use_pca = use_pca
        self.n_components = n_comp
#        self.shift_predict = shift_predict
        super(classifier, self).__init__( *args, **kwds)

    def fit(self, pfds, target):
        """
        args: pfds, target
        pfds: the training pfds
        target: the training targets
        """
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

    def predict(self, pfds, shift_predict=False):
        """
        args: pfds, target
        pfds: the testing pfds
        """
#        if shift_predict is None:
#            shift_predict = self.shift_predict

        data = np.array([pfd.getdata(**self.feature) for pfd in pfds])
        #self.test_data = data
        current_class = self.__class__
        self.__class__ = self.orig_class
        if shift_predict:
            shift_probs = []
            nbins = self.feature.values()[0]
            nsamples = data.shape[0]
            for shift in range(nbins):
                if self.feature.keys()[0] == 'phasebins':
                    shift_data = np.roll(data, shift, axis=1)
                else:
                    rdata = data.reshape((nsamples, nbins, nbins))
                    shift_data = np.roll(rdata, shift, axis=2).reshape((nsamples,nbins*nbins))
                if self.use_pca:
                    shift_data = self.pca.transform(shift_data)
                    #AAR: not compatible with multi-class (future fix)
                shift_probs.append(self.predict(shift_data))
            results = shift_probs #(nbins, Nsamples)
        else:
            if self.use_pca:
                data = self.pca.transform(data)
            results =  self.predict(data)

        self.__class__ = current_class
        return results
        #return self.orig_class.predict(self, data)
        
    def predict_proba(self, pfds, shift_predict=False):
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
        Returns the probability of the sample for class '1' in
        the model. Not compatible with multi-class yet.
    if shift_predict:
    Returns X : list-like, shape = [nbins, n_samples]
                Returns the probability of class 1 in the model
                at each phase. Processed in combinedAI to determine
                best phase.
    Notes
    -----
        """
#        if shift_predict is None:
#            shift_predict = self.shift_predict

        data = np.array([pfd.getdata(**self.feature) for pfd in pfds])
        current_class = self.__class__
        self.__class__ = self.orig_class
        if shift_predict:
            shift_probs = []
            nbins = self.feature.values()[0]
            nsamples = data.shape[0]
            for shift in range(nbins):
                if self.feature.keys()[0] == 'phasebins':
                    shift_data = np.roll(data, shift, axis=1)
                else:
                    rdata = data.reshape((nsamples, nbins, nbins))
                    shift_data = np.roll(rdata, shift, axis=2).reshape((nsamples,nbins*nbins))
                if self.use_pca:
                    shift_data = self.pca.transform(shift_data)
                    #AAR: not compatible with multi-class (future fix)
                shift_probs.append(self.predict_proba(shift_data)[...,1])
            results = shift_probs #(nbins, Nsamples)
        else:
            if self.use_pca:
                data = self.pca.transform(data)
            #AAR: not compatible with multi-class (future fix)
            results =  self.predict_proba(data)[...,1]

        self.__class__ = current_class
        return results

    def score(self, pfds, target, F1=True):
        """
        args: pfds, target
        pfds: the testing pfds
        target: the testing targets
        """
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

    def plot_shiftpredict(self, pfd, compare=None):
        """
        plot the predictions vs phase-shift for this classifier
        Args:
        pfd = a single pfddata object
        compare : an array of Prob(phase) that is plotted for comparison
     
        saves output to file based on class and shift
        """
        import pylab as plt
        curclass = self.__class__
        self.__class__ = self.orig_class

        data = np.array(pfd.getdata(**self.feature))

        nbin = self.feature.values()[0]
        feature = self.feature.keys()[0]
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
            if self.use_pca:
                sdata = self.pca.transform(sdata.flatten())
            if D == 1:
                preds.append(self.predict_proba([sdata])[...,1][0])
            else:
                preds.append(self.predict_proba([sdata.flatten()])[...,1][0])

        for shift in range(nbin):
            fout = "%s_%s%i-%03d" %\
                (str(type(self)).split('.')[-1].strip('>').strip("'"), feature, nbin, shift)
            plt.clf()
            plt.subplots_adjust(hspace=0.15)
            sdata = np.roll(data, shift, axis=D-1)

            # show Prob(phase), orig data(phase), pca data(phase)
            if self.use_pca:
                pdata = self.pca.inverse_transform(self.pca.transform(sdata.flatten()))

                ax1 = plt.subplot2grid((2,2), (0,0), colspan=2)#, aspect='equal')
                ax1.plot(x, preds, 'b',label='%s' % \
                             str(type(self)).split('.')[-1].strip('>').strip("'"))
                ax1.plot(x[shift], preds[shift], 'bo', markersize=10, alpha=0.5)
                if compare is not None:
                    ax1.plot(x, compdata, 'r',label='combinedAI')
                    ax1.plot(x[shift], compdata[shift], 'ro', markersize=10, alpha=0.5)
                ax1.set_ylabel('Probability')
                ax1.set_title('%s, %s, shift %i' % \
                                  (str(type(self)).split('.')[-1].strip('>').strip("'"),
                                   self.feature, shift))
                ax1.set_ylim(0, 1)
                ax1.set_xlabel('Phase Shift')
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
                    ax2.plot(x, pdata)
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
                if compare is not None:
                    ax1.plot(x, compdata, 'r',label='combinedAI')
                    ax1.plot(x[shift], compdata[shift], 'ro', markersize=10, alpha=0.5)
                ax1.set_ylabel('Probability')
                ax1.set_title('%s, %s, shift %i' % \
                                  (str(type(self)).split('.')[-1].strip('>').strip("'"),
                                   self.feature, shift))
                ax1.set_ylim(0,1)
                plt.setp( ax1.get_xticklabels(), visible=False)

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

        self.__class__ = curclass


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



