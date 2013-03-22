"""
Aaron Berndsen:
class which wraps combinedAI with a 'feature'.

"""
import numpy as np
from ubc_AI.data import pfdreader
from sklearn import linear_model, svm, mixture
from scipy.optimize import curve_fit
import scipy.stats as stats

# Define model function to be used to fit to the data above:
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))
def gaussfit(data):
    """given a some data, return the stddev"""
    hist, bin_edges = np.histogram(data, density=True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
    # p0 is the initial guess for the fitting coefficients
    # (A, mu and sigma above)
    p0 = [1., 0., 1.]
    try:
        coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)
    except(RuntimeError):
        coeff = np.ones(3)
        var_matrix = np.eye(3)
    return coeff

class cAIcAI(object):
    def __init__(self, cAI, AIonAI='lr', feature={'phasebins':32}):
        """
        a way to combine the prediction matrix of a combinedAI
        with a feature from a pfd.
        
        args:
        cAI : combinedAI object
        AIonAI : combine cAI with feature using this algorithm
                 ['lr', 'svm']
        """
        self.cAI = cAI
        self.feature = feature
        if AIonAI == 'lr':
            self.AIonAI = linear_model.LogisticRegression()
        else:
            self.AIonAI = svm.SVC()
        # define the gaussian-fitting routine here: takes in the feature data
        # and returns a 1d array
        self.GF = gaussfit
        #self.GF = stats.norm.fit
    def fit(self, X, y):
        """
        X : list of pfds
        y : targets
        Note, y must be list of pfdreader objects
        """
        self.cAI.fit(X,y)
        preds = self.cAI.predict(X, pred_mat=True)
        feats = np.array([self.GF(pfd.getdata(**self.feature)) for pfd in X])
        data = np.hstack([preds, feats])
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if y.ndim == 2:
            target = y[:,0]
        else:
            target = y
        self.AIonAI.fit(data, target)

    def predict(self, X):
        preds = self.cAI.predict(X, pred_mat=True)
        feats = np.array([self.GF(pfd.getdata(**self.feature)) for pfd in X])
        data = np.hstack([preds, feats])
        return self.AIonAI.predict(data)
    
    def predict_proba(self, X):
        preds = self.cAI.predict(X, pred_mat=True)
        feats = np.array([self.GF(pfd.getdata(**self.feature))for pfd in X])
        data = np.hstack([preds, feats])
        return self.AIonAI.predict_proba(data)
