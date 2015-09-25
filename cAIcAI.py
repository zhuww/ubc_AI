"""
Aaron Berndsen:
class which wraps combinedAI with a 'feature'.

"""
import numpy as np
from ubc_AI.data import pfdreader
from sklearn import linear_model, svm, mixture
from scipy.optimize import curve_fit
import scipy.stats as stats

# needed to extract harmonic of 60 Hz information:
from ubc_AI.prepfold import pfd as PFD
import fractions

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
def PF0_fit(data, harms):
    """ given a distribution of survey F0's, and list of harmonics,
    fit the PDF as 'uniform plus \sum_harmonics Normal'
    Return [A]*Nharm

    """
    As = np.ones_like(harms)
    p = [As, harms]
    coeff, var_matrix = curve_fit(gauss, data, )
def PF0_gauss(x, *p):
    A, mu = p
    return sum(A*np.exp(-(x-mu)**2/(2.*.1)**2))

def harm_ratio(f0, f=60., max_denom=500):
    """
    given a frequency 'f0', compare to 'f' [default 60] and
    return the %-difference of the nearest f0 harmonic to f (default 60 Hz)

    """
    c = fractions.Fraction(f0/f).limit_denominator(max_denominator=max_denom)
    if c.numerator == 0:
        c = fractions.Fraction(f/f0).limit_denominator(max_denominator=max_denom)
        diff = (f - f0*c.numerator/c.denominator)/f
    else:
        diff = (f - f0*c.denominator/c.numerator)/f
    return diff

class cAIcAI(object):
    def __init__(self, cAI, AIonAI='lr', feature={'phasebins':32}):
        """
        a way to combine the prediction matrix of a combinedAI
        with a feature from a pfd.
        
        args:
        cAI : combinedAI object
        AIonAI : combine cAI with feature using this algorithm
                 ['lr', 'svm']
        feature : one of the combinedAI features
                 or 
                 {'60hz':max_denom) : append harmonic difference of 
                                 candidate to 60Hz as af eature
                 or
                 {'hist':array} : supply an array of [F0's] for all candidates.
                        in this case 'predictt' and 'predict_proba' return
                        P(p|F0) = P(F0|p) * P(p) / P(F0)
                        P(F0|p) is an exponential
                        P(F0) is determined by the array
                 
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
        if '60hz' in self.feature:
            md = self.feature['60hz']
            feats = np.array([harm_ratio(1./PFD(pfd.pfdfile).topo_p1, max_denom=md)\
                                  for pfd in X])
        elif ['hist'] in self.feature:
            h = np.array(self.feature['hist'])
            # fit the histogram as a uniform distribution plus 
            # sum of guassians on each harmonic (of width .1 MHz)
            low = h.min()
            hgh = h.max()
            bins = np.arange(low, hgh, .2) #.2 Hz bins
            H  = np.histogram(h, bins=bins)
            harms = []
            for i in range(99):
                harms.append(fractions.Fraction(i+1, 100)*60.)
                harms.append(fractions.Fraction(100, i+1)*60.)
                                  
            f = 1./(hgh - low) #+ gauss(x, (XX
            
        else:
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
        if self.feature != '60hz':
            feats = np.array([self.GF(pfd.getdata(**self.feature)) for pfd in X])
        else:
            feats = np.array([harm_ratio(1./PFD(pfd.pfdfile).topo_p1)\
                                  for pfd in X])
        data = np.hstack([preds, feats])
        return self.AIonAI.predict(data)
    
    def predict_proba(self, X):
        preds = self.cAI.predict(X, pred_mat=True)
        if self.feature != '60hz':
            feats = np.array([self.GF(pfd.getdata(**self.feature))for pfd in X])
        else:
            feats = np.array([harm_ratio(1./PFD(pfd.pfdfile).topo_p1)\
                                  for pfd in X])            
        data = np.hstack([preds, feats])
        return self.AIonAI.predict_proba(data)
