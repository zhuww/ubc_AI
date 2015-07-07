from ubc_AI.threadit import threadit
#threadit.func_defaults[0]['state'] = True
from ubc_AI.training import pfddata

from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import ubc_AI.pulsar_nnetwork as pnn
import ubc_AI.training
from ubc_AI.data import dataloader
import ubc_AI.classifier as CLF
import numpy as np
import time
import cPickle
import copy
from ubc_AI.data import singleclass_score


"""
load data
Class dataloader() will load data from either a pickled dataloader object or a text file
with a list of pfd files and their classifications. The classifications should follow the
pfd file name(with path) and can be either in 1 column format or 5 column format. In 5 
column format, the classifications should be in the order:
pulsar | pulse profile | DM curve | time-vs-phase | frequency-vs-phase
"""
#ldf = dataloader('Broadfeature.pkl')
#ldf = dataloader('Broadfeature_1.txt')
#ldf = dataloader('Narrowfeature_1.txt')
ldf = dataloader('ExtraBroad.txt')
print "Running"
#for p in ldf.pfds:
    #p.extracted_feature = {}
    #p.__init__(p.pfdfile)
#print 'data loaded'
#ldf.split()
#tnames = ['rfi', 'pulsar']

"""define the best clfs:"""
nn1 = CLF.pnnclf(design=[25], gamma=0.5, feature={'phasebins':64},maxiter=None) #F1 .74
nn2 = CLF.cnnclf(feature={'intervals':48}, poolsize=[(3,3),(2,2)],n_epochs=65, batch_size=20, nkerns=[20,40], filters=[16,8], L1_reg=1., L2_reg=1.)
nn3 = CLF.pnnclf(design=[9], gamma=0.001, use_pca=True, n_comp=24,
                  feature={'timebins':64}, maxiter=None)                         #F1 .83
nn4 = CLF.cnnclf(feature={'subbands':48}, poolsize=[(3,3),(2,2)],n_epochs=65, batch_size=20, nkerns=[20,40], filters=[16,8], L1_reg=1., L2_reg=1.)
nn5 = CLF.pnnclf(design=[9], gamma=0.1,feature={'DMbins':60}, maxiter=None)       #F1 .80
clf1 = CLF.svmclf(gamma=0.05, C=1.0, feature={'phasebins':64}, probability=True)
clf2 = CLF.svmclf(gamma=0.005, C=5, feature={'intervals':64}, use_pca=True, n_comp=24, probability=True)
clf3 = CLF.svmclf(gamma=0.001, C=24., feature={'subbands':64}, use_pca=True, n_comp=24, probability=True)
clf4 = CLF.svmclf(gamma=0.2, C=25., feature={'DMbins':60}, probability=True)
clf5 = CLF.svmclf(gamma=85., C=1.0, feature={'timebins':52})
lg1 = CLF.LRclf(C=32., penalty='l2', use_pca=True, n_comp=32, feature={'subbands':52})                           #F1 .79
lg2 = CLF.LRclf(C=50., penalty='l2', use_pca=True, n_comp=24, feature={'intervals':64})                          #F1 .80
lg3 = CLF.LRclf(C=0.07, penalty='l2', feature={'DMbins':60})
tree3 = CLF.dtreeclf(min_samples_split=8, min_samples_leaf=4, feature={'DMbins':60})
    #other features performed poorly for LG
"""code for testing the performance of individual layer-1 classifiers"""
#ldf.cross_val_score(clf1, verbose=False)
#ldf.cross_val_score(clf2, verbose=False)
#ldf.cross_val_score(clf3, verbose=False)
#ldf.cross_val_score(clf4, verbose=False)
#ldf.cross_val_score(nn1, verbose=False)
#ldf.cross_val_score(nn2, verbose=False)
#ldf.cross_val_score(nn4, verbose=False)
#ldf.cross_val_score(nn5, verbose=False)

""" join the layer-1 classifiers into a combined-AI """
AIs = [nn1, nn2, nn3, nn4, nn5, clf1, clf2, clf3, clf4,  clf5, tree3, lg1, lg2]
#good combos discovered so far:
combo = [0, 1, 3, 4, 5, 6, 7, 8] #F1 = 0.9
#combo = [ 5, 1, 3, 8] #F1 = 0.9
#combo = [0, 1, 3, 4] #F1 = 0.9
#combo = [5, 6, 7, 8] #F1 = 0.9

#remapper = "lambda x:1./(1/x/99. + 98./99.)"
#remapper = "1./(1/%s/99. + 98./99.)"

clfl2 = CLF.combinedAI([AIs[i] for i in combo], strategy='lr', C=0.1, penalty='l2')
#print 'l2:', ldf.cross_val_score(clfl2, cv=5, verbose=True) #evaluate F1
#ldf.learning_curve(clfl2) #plot learning curve
#show()

#clfsvm = CLF.combinedAI([AIs[i] for i in combo], strategy='lr', C=1., penalty='l2')
#print 'svm:', ldf.cross_val_score(clfsvm, verbose=False)

"""training and saving the combined AI"""
clfl2.fit(ldf.pfds, ldf.target)
cPickle.dump(clfl2, open('clfl2_new.pkl' ,'wb'), protocol=2)

