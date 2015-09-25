"""
A moudule for the new datafitter class that works with the new classifier class.
"""
import numpy as np
from random import shuffle
import cPickle
from scipy import mgrid
import os,sys
from ubc_AI.training import pfddata
from ubc_AI.psrarchive_reader import ar2data
from ubc_AI.singlepulse import singlepulse
from ubc_AI.singlepulse import SPdata

class pfdreader(object):
    """ 
    A new pfd reader class that only store the link to the file and the extracted data.
    """
    SearchPATH = "/home/zhuww/work/AI_PFD/training/PFDfiles/pulsars/:/home/zhuww/work/AI_PFD/training/PFDfiles/RFIs/:/home/zhuww/work/AI_PFD/training/PFDfiles/nonpulsars/:/home/zhuww/work/AI_PFD/training/PFDfiles/harmonics/"
    def __init__(self, pfdfile):
        #search for the file
        self.extracted_feature = {}
        if pfdfile.__class__ == pfddata:
            self.extracted_feature.update(pfdfile.extracted_feature)
            for cls in ['PSRclass', 'SmPclass', 'DMCclass', 'TvPclass', 'FvPclass']:
                if cls in pfdfile.__dict__:
                    self.__dict__.update({cls:pfdfile.__dict__[cls]})
            pfdfile = pfdfile.pfd_filename
        elif pfdfile.__class__ == singlepulse:
            self.extracted_feature.update(pfdfile.extracted_feature)
            self.pfdfile = pfdfile
        else:
            if os.access(pfdfile, os.R_OK):
                self.pfdfile = pfdfile
            else:
                for path in self.SearchPATH.split(':'):
                    if os.access(path+pfdfile, os.R_OK):
                        self.pfdfile = path+pfdfile
                        break
                    elif os.access(path+pfdfile.split('/')[-1], os.R_OK):
                        self.pfdfile = path+pfdfile.split('/')[-1]
                        break
                if not 'pfdfile' in self.__dict__:
                    #print pfdfile, self.PSRclass, self.DMCclass
                    raise NameError, "did not find the file %s" % pfdfile

    def getdata(self, *fargs, **features):
        pfd = None
        def extract(key, value, pfd):
            feature = '%s:%s' % (key, value)
            if feature in self.extracted_feature:
                #print 'use extracted feature %s' % feature
                newdata = self.extracted_feature[feature]
            else:
                #print 'extracting new feature %' % feature
                newdata = pfd.getdata(**{key:value})
                self.extracted_feature.update({feature:newdata})
            return newdata
        
        data = np.array([])
        #process the args (a list of single-item dictionaries)
        for i in fargs:
            key, value = i.items()[0]
            feature = '%s:%s' % (key, value)
            if (feature not in self.extracted_feature) and (pfd is None):
                if not type(self.pfdfile) is str and self.pfdfile.__class__ == singlepulse:
                    pfd = self.pfdfile
                elif os.path.splitext(self.pfdfile)[1] == '.pfd': 
                    pfd = pfddata(self.pfdfile, align=True) 
                elif os.path.splitext(self.pfdfile)[1] == '.ar2':  
                    pfd = ar2data(self.pfdfile, align=True) 
                elif os.path.splitext(self.pfdfile)[1] == '.ar':  
                    pfd = ar2data(self.pfdfile, align=True) 
                elif os.path.splitext(self.pfdfile)[1] == '.spd':  
                    pfd = SPdata(self.pfdfile, align=True) 
                else:
                    print "unrecognized file format ", self.pfdfile
                    raise Error
            data = np.append(data, extract(key, value, pfd))
        #process the kwargs
        for key, value in features.iteritems():
            feature = '%s:%s' % (key, value) 
            if (feature not in self.extracted_feature) and (pfd is None):
                if not type(self.pfdfile) is str and self.pfdfile.__class__ == singlepulse:
                    pfd = self.pfdfile
                elif os.path.splitext(self.pfdfile)[1] == '.pfd': 
                    pfd = pfddata(self.pfdfile, align=True) 
                elif os.path.splitext(self.pfdfile)[1] == '.ar2':  
                    pfd = ar2data(self.pfdfile, align=True) 
                elif os.path.splitext(self.pfdfile)[1] == '.ar':  
                    pfd = ar2data(self.pfdfile, align=True) 
                elif os.path.splitext(self.pfdfile)[1] == '.spd':  
                    pfd = SPdata(self.pfdfile, align=True) 
                else:
                    print "unrecognized file format ", self.pfdfile
                    raise Error
            data = np.append(data, extract(key, value, pfd))
        del(pfd)
        return data

def singleclass_score(classifier, test_pfds, test_target, verbose=False):
    pulsar = set([])
    truepulsar = set([])
    pred_prob = classifier.predict_proba(test_pfds)[...,1]
    pred = np.where(pred_prob>0.5, 1, 0)
    if not test_target.ndim == 1:
        try:
            feature_target = test_target[..., classifier.targetmap[classifier.feature.keys()[0]]]
        except AttributeError:
            feature_target = test_target[..., 0]
    else:
        feature_target = test_target

    for i,p in enumerate(pred):
        #print test_target[i], int(predict)
        if int(feature_target[i]) == 1:
            truepulsar.add(i)
        if  int(p) == 1:
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





def cross_validation(classifier, pfds, target, cv=5, verbose=False):
    #classifier = classifier()
    nclasses = len(np.unique(target))
    if verbose:cv = 1
    scores = np.array([])
    arglists = []
    for i in range(cv):
        L = len(pfds)
        pfds = np.array(pfds)
        index = range(L)
# keep shuffling until training set has all types
        while 1:
            shuffle(index)
            cut = int(0.6*L)
            training_idx = index[:cut]
            test_idx = index[cut:]

            training_pfds = pfds[training_idx]
            training_target = target[training_idx]
            test_pfds = pfds[test_idx]
            test_target = target[test_idx]
            if len(np.unique(training_target)) == len(np.unique(target)):
                break
        n_samples = len(training_pfds)
        #training_pfds = training_pfds.reshape((n_samples, -1))
        #classifier = svm.SVC(gamma=0.1, scale_C=False)
        arglists.append([classifier, training_pfds, training_target, test_pfds, test_target])
        #classifier.fit(training_pfds, training_target)

    def getF1(clf, training_pfds, training_target, test_pfds, test_target):
        clf.fit(training_pfds, training_target)
        F1 = singleclass_score(clf, test_pfds, test_target, verbose=verbose)
        return F1

    if not nclasses == 2:
        raise "not yet implemented multiclass_score"
        #F1 = multiclass_score(classifier, test_pfds, test_target,
                              #nclasses = nclasses, verbose=verbose)
    else:
        #F1 = singleclass_score(classifier, test_pfds, test_target, verbose=verbose)
        #if classifier.__dict__.has_key('strategy'):
            #F1dict = dict([(i,getF1(*al))for i,al in enumerate(arglists)])

        from ubc_AI.threadit import threadit
        if len(arglists) >= 12:
            F1dict = threadit(getF1, arglists)
        else:
            F1dict = dict([(i,getF1(*al))for i,al in enumerate(arglists)])
    #scores = np.append(scores, F1)
    #print F1dict
    scores = np.array([F1dict[i] for i in F1dict])

    return scores





class dataloader(object):
    """
    A class to hold the data and provide methods for testing AIs.

    """

    def __init__(self, filename, classmap=None):
        """
        initialize from a filename, to create a Datafitter instance 
        that holds the data and perform fitting using provided classifier
        args: filename, classmap
        filename: the name of the pickle file
        classmap: mapping for different classes
        """
        self.trainclassifiers = {}
        if filename.endswith('.pkl'):
            with open(filename, 'r') as fileobj:
                originaldata = cPickle.load(fileobj)
                self.pfds = originaldata['pfds']
                if type(originaldata['target']) in [list] or originaldata['target'].ndim == 1:
                    self.orig_target = originaldata['target']
                    if classmap == None:
                        self.classmap = {0:[4,5], 1:[6,7]}
                    else:
                        self.classmap = classmap
                    self.target = self.orig_target[:]
                    for k, v in self.classmap.iteritems():
                        for val in v:
                            self.target[self.orig_target == val] = k 
                else:
                    self.target = originaldata['target']
        elif filename.endswith('.txt'):
            data = np.loadtxt(filename, dtype=[('fname', '|S200'), ('Overall', int), ('Profile', int), ('Interval', int), ('Subband',int), ('DMCurve', int)], comments='#')
            self.pfds = [ pfdreader(f) for f in data['fname']]
            self.target = np.vstack((data['Overall'], data['Profile'], data['DMCurve'], data['Interval'], data['Subband'])).T

        else:
            print "Don't recognize the file surfix."
            raise Error
        self.extracted_feature = []

    def extractfeatures(self, clf):
        if type(clf) == list:
            AIlist = clf
        elif 'list_of_AIs' in clf.__dict__:
            AIlist = clf.list_of_AIs
        elif 'feature' in clf.__dict__:
            AIlist = [clf] 
        else:
            raise MyError
        features = {}
        vargf = []
        items = []
        for clf in AIlist:
            items.extend(clf.feature.items())
        for f in set(items):
            if not f in self.extracted_feature:
                vargf.append(dict([f]))
        def getfeature(pfd):
            pfd.getdata(*vargf, **features)
            return pfd
        from ubc_AI.threadit import threadit
        if len(vargf) > 0:
            resultdict = threadit(getfeature, [[p] for p in self.pfds])
            for n, pfd in resultdict.iteritems():
                self.pfds[n] = pfd
        for f in vargf:
            self.extracted_feature.append(f)


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

    def split(self, pct=0.6):
        """
        Given some complete set of pfds and their targets,
        split the indices into 'pct' training, '1-pct' cross-vals
        
        Args:
        target = data classifications
        pct = 0 < pct < 1, default 0.6

        returns:
        training_data, training_target, test_data, test_target

        """
        from random import shuffle
        if isinstance(self.pfds,type([])):
            pfds = np.array(self.pfds)
        target = self.target

        L = len(target)
        index = range(L)
        cut = int(pct*L)
        while 1:
            shuffle(index)
            train_idx = index[:cut]
            self.train_target = target[train_idx]
            self.train_pfds = pfds[train_idx]

            test_idx = index[cut:]
            self.test_target = target[test_idx]
            self.test_pfds = pfds[test_idx]
            
    # make sure training has samples from all classes
            if len(np.unique(self.train_target)) == len(np.unique(self.target)):
                break

        self.trainclassifiers = {}

        
    def train(self, clf):
        """
        train the classifier
        args:; classifier created using the mixin classifier class
        """
        if not 'test_pfds' in self.__dict__ or not 'test_target' in self.__dict__:
            self.split()
        self.trainclassifiers[clf] = True 
        clf.fit(self.train_pfds, self.train_target)


    def cross_val_score(self, classifier, cv=5, verbose=False):
        """
        calculate the cross validation score
        input: classifier, cv, verbose=False
        cv: number of trails
        verbose: if True than print out recall, precision and more.
        """
        #L = len(self.data[0])
        #classifier = clsFunc(L)
        scores = cross_validation(classifier, self.pfds, self.target, cv=cv, verbose=verbose)
        print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
        return scores

    def learning_curve(self, classifier,
                       pct=0.6,
                       plot=True):
        """
        plot the learning curve, error vs training data size, to see if it is necessary to include more training data.
        
        args: classifier, plot=True, pct=0.6
        """
        self.split()
        X = self.train_pfds
        y = self.train_target
        Xval = self.test_pfds
        yval = self.test_target
        m = y.shape[0]
        stepsize = max(m/25,1)
        ntrials = range(1,m,stepsize)
        mm = len(ntrials)
        t_F1 = np.zeros(mm)
        v_F1 = np.zeros(mm)
        for i, v in enumerate(ntrials):
            #fit with regularization
            classifier.fit(X[0:v+1], y[0:v+1])
            
            # but compute F1 without regularization
            t_F1[i] =  classifier.score(X[0:v+1], y[0:v+1])
            # use entire x-val set
            v_F1[i] =  classifier.score(Xval, yval)
            
        if plot:
            import matplotlib.pyplot as plt
            plt.plot(ntrials, t_F1, 'r+', label='training')
            plt.plot(ntrials, v_F1, 'bx', label='x-val')
            plt.xlabel('training set size')
            plt.ylabel('F1')
            plt.legend()
            plt.show()
            return None
        else:
            return t_F1, v_F1, ntrials

    def feature_curve(self, classifier, 
                      feature = None,
                      bounds=None,
                      Npts=10,
                      plot=True,
                      pct=0.6):
        """
        plot the feature curve, error vs feature size, to probe what is the best size to choose for a certain feature.
        
        args: classifier, feature={'intervals':32}, bounds=[8,32], Npts=10, plot=True, pct=0.6
        """
        pfds = self.pfds
        target = self.target
        if bounds == None:
            vals = mgrid[8:32:1j*Npts]
        else:
            vals = mgrid[bounds[0]:bounds[1]:1j*Npts]

        if feature == None:
            feature = classifier.feature.keys()[0]
        train_score = np.zeros_like(vals)
        test_score = np.zeros_like(vals)
        for i, val in enumerate(vals):
            classifier.feature[feature] = int(val)
            self.split()
            self.train(classifier)
            train_score[i] = 1-classifier.score(self.train_pfds, self.train_target)
            test_score[i] = 1-classifier.score(self.test_pfds, self.test_target)
        if plot:
            import matplotlib.pyplot as plt
            plt.plot(vals, train_score, 'r+', label='training')
            plt.plot(vals, test_score, 'bx', label='x-val')
            plt.xlabel(feature)
            plt.ylabel('error')
            plt.legend()
            plt.show()
        else:
            return train_score, test_score, vals, vals[test_score.argmax()]

    def PR_curve(self, clf, Pcut=None):
        """
        Plot the precision vs recall curve, recall vs P-cut, precision vs P-cut, F1 vs P-cut curves. Only works when output probability is turned on.
        input: classifier
               Pcut = np.arange(0.05, 1.0 0.05)
        """
        self.split()
        clf.fit(self.train_pfds, self.train_target)
        #predict = clf.predict(self.test_pfds)
        target = self.test_target
        Proba = clf.predict_proba(self.test_pfds)
        if Pcut == None:
            Pcut = np.arange(0.05,1.0,0.05)
        P = []
        R = []
        F1 = []
        for cut in Pcut:
            predict = np.where(Proba >= cut, 1, 0)
            p = np.mean(target[predict == 1])
            r = np.mean(predict[target == 1])
            P.append(p)
            R.append(r)
            F1.append(2 * p * r / (p + r))
        import matplotlib.pyplot as plt
        plt.figure(figsize=(2,2))
        ax = plt.subplot(221)
        ax.plot(P, R, '-')
        ax.set_xlabel('Precision')
        ax.set_ylabel('Recall')
        ax = plt.subplot(222)
        ax.plot(Pcut, R, '-')
        ax.set_xlabel('Probability cut')
        ax.set_ylabel('Recall')
        ax = plt.subplot(223)
        ax.plot(Pcut, P, '-')
        ax.set_xlabel('Probability cut')
        ax.set_ylabel('Precision')
        ax = plt.subplot(224)
        ax.plot(Pcut, F1, '-')
        ax.set_xlabel('Probability cut')
        ax.set_ylabel('F1')
        plt.show()





    def plot_prediction(self, clf, what, feature=None, plot=True):
        """
        plot misses, false positives, pulsars, and recommendations
        args: classifier, what_to_plot, feature={'intervals':32}, plot=True
        what_to_plot: takes value in ['miss', 'falsepos', 'truepulsar', 'pulsar']
        if plot == True: show plot
        else: return the indics of the chosen pfds
        """
        if not 'test_pfds' in self.__dict__ or not 'test_target' in self.__dict__:
            self.train(clf)
        elif not clf in self.trainclassifiers:
            clf.fit(self.train_pfds, self.train_target)

        if feature == None:
            self.kwds = {'intervals':32}
        else:
            self.kwds = feature

        pdts =  clf.predict(self.test_pfds)
        truepulsar = set([])
        pulsar = set([])
        if self.test_target.ndim > 1:
            mytarget = self.test_target[...,0]
        else:
            mytarget = self.test_target
        for i,p in enumerate(pdts):
            if int(mytarget[i]) == 1:
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

        test_data = [pf.getdata(**self.kwds) for pf in self.test_pfds]

        if plot:
            import matplotlib.pyplot as plt
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
                            ax.imshow(test_data[what[i]].reshape(N,N))
                        else:
                            ax.plot(test_data[what[i]])
                    except IndexError:pass
                    i += 1
                    ax.set_yticklabels([])
                    ax.set_xticklabels([])

            plt.show()
        else:
            return what


    def plot_samples(self, feature=None, sample_list=[], testonly=False):
        """
        plot the list of samples, given a list of
        their index numbers

        Args:
        sample_list = list of sample indices of self.pdfs to plot (maximum 64)
        feature: the feature to extract, default is {'intervals':32}
        """
        if feature == None:
            self.kwds = {'intervals':32}
        else:
            self.kwds = feature
        if isinstance(sample_list,type(set([]))):
            sample_list = list(sample_list)
        if testonly:
            test_data = [pf.getdata(**self.kwds) for pf in self.test_pfds]
        else:
            test_data = [pf.getdata(**self.kwds) for pf in self.pfds]

        import matplotlib.pyplot as plt
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
                        ax.imshow(test_data[sample_list[i]].reshape(N,N), cmap=plt.get_cmap("binary"))
                                  #cmap=plt.cmap.gray)
                    else:
                        ax.plot(test_data[sample_list[i]])
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
            p = clf.predict(self.test_pfds)
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
