#!/usr/bin/env python
"""
Aaron Berndsen (2012)

a neural network implementation in python, with f2py/fortran 
optimizations for large data sets

"""
import numpy as np
import cPickle
import pylab as plt
from scipy import io
from scipy import mgrid
from scipy.optimize import fmin_cg
import sys
from sklearn.base import BaseEstimator

#Aaron's fortran-optimized openmp code
try:
    from nnopt import sigmoid2d, matmult
    _fort_opt = True
except(ImportError):
    print "NeuralNetwork not using fortran-optimized code"
    print "Refer to nn_optimize.f90 for tips"
    print "defaulting to vectorized numpy implementation"
    _fort_opt = False

#number of iterations in training
_niter = 0

def main():
    """ not really used """
################
# load the data 
    data = handwritingpix(samples='ex4data1.mat',
                          thetas='ex4weights.mat'                              
                          )
################
# show some samples of the data
    data.plot_samples(15)
###############
# create the neural network, 
    nn = NeuralNetwork(gamma=0.)

def load_pickle(fname):
    """
    recover from a pickled NeuralNetwork classifier

    the neural design (nfeatures, nlayers, ntargets) are 
    determined by the 'thetas' in the pickle 

    """
    d = cPickle.load(open(fname,'r'))
    nlayers = d['nlayers']
    gamma = d['gamma']
    thetas = []
    for li in range(nlayers):
        thetas.append(d[li])

    classifier = NeuralNetwork(gamma=gamma, thetas=thetas)
    return classifier


class layer(object):
    """
    a layer in the neural network. It takes 
    in N parameters and outputs m attributes

    N and m should include the bias term (if appropriate)
    
    output = Theta*trial_colvector

    optional:
    theta = the transformation matrix of size (N+1,m) (includes bias)
          otherwise randomly initialize theta
          we uniformly initialize theta over [-delta, delta]
          where delta = sqrt(6)/sqrt(N+m), or passed as argument
    shiftlayer = False. If true, we repeat the first neuron weightings, shifted.

    """
    def __init__(self, n, m, theta=np.array([]), delta=0, shiftlayer=False):
        #n includes bias
        self.inp = n
        self.output = m
        self.shiftlayer = shiftlayer
        if len(theta):
            self.theta = theta
        else:
            # use random initialization
            if not delta:
                delta = np.sqrt(6)/np.sqrt(n + m)
            self.randomize(delta)

    def randomize(self, delta=None):
        """
        randomize the theta's for each 'fit' call in the neural network
        """
        N = self.inp
        m = self.output
        if not delta:
            delta = np.sqrt(6)/np.sqrt( N + m)
        self.theta = np.random.uniform(-delta, delta, N*m).reshape(N, m)

        if self.shiftlayer:
            #make sure the subsequent layers are simply a shifted copy of the first
            l1 = self.theta[:,0]
            dshift = N//m
            for i in range(m-1):
                shift = dshift * (i+1)
                self.theta[:,i+1] = np.roll(l1, shift)

class NeuralNetwork(BaseEstimator):
    """
    a collection of layers forming the neural network

    Args:
    * gamma = learning rate (regularization parameter), default = 0.
    * thetas = None (initialize the NN with neurons/layers determined
              by the list of internal 'Thetas'
              (default = None, defers until .fit call)
    * design = None (initialize the NN with neurons/layers determined
              by the list of neurons per layer
              Eg. desing=[25,4] = 2-layers, first with 25, second with 4 
    * fit_type = ['all','single'] . Fit the individual layers 'all' at once (default)
                              or build up the NN, fitting a 'single' layer at a time,
                              then adding the next layer.
    * maxiter : number of iterations in self.fit's conjugate-gradient minimization
                default = 100, can be overriden elsewhere (self.fit)
    * shiftlayer : if not None, instead of having N independent neurons in this layer,
                 we reproduce the first neuron N-times, shifting the weights of the first one.
                 This is meant to help remove phase-dependence of the pulse.
                 shiftlayer in [0, 1, 2, ...] number of hidden layers
    
    Notes: 
    * if design != None and thetas != None, we get shape
      from the thetas

    * if design = thetas = None, we determine design in fit routine

    """
    def __init__(self, gamma=0., thetas=None, design=None,
                 fit_type='all', maxiter=None, shiftlayer=None, verbose=False):
        self.gamma = gamma
        self.design = design
        self.fit_type = fit_type
        self.nin = None
        self.nout = None
        if maxiter == None:
            self.maxiter = 100
        else:
            self.maxiter = maxiter
        self.shiftlayer = shiftlayer
        if thetas != None:
            nfeatures = thetas[0].shape[0]-1
            ntargets = thetas[1].shape[1]
            self.create_layers(nfeatures, ntargets, thetas=thetas, verbose=verbose,\
                                   shiftlayer=shiftlayer)
        self.verbose = verbose
        self.nfit = 0 # keep track of number of times the classifier has been 'fit'

    def create_layers(self, nfeatures, ntargets, design=None, gamma=None,
                      thetas=None, verbose=None, shiftlayer=None):
        """
        This routine is called by 'fit', and adjust the network design
        for varying feature and target length, as well as network design.

        Args:
        nfeatures: number of features
        ntargets : number of target labels
        design : list of neurons per layer. 
             Default = None, use self.design (if provided at init), 
                 otherwise default to [16] = one layer of 16 neurons
            Eg. design=[16,4] = a layer of 16 neurons, then a layer of 4 neurons
        gamma : update self.gamma, the regularization parameter
        thetas = None : can pass neural mappings as list of arrays (a list of thetas),
                        otherwise they are randomly initialized (better).
                      This overrides 'design'
        shiftlayer = None: if not None, the theta is simply a shifted repeat of the first neuron
                           in this layer
                   
        """
        if design == None:
            if self.design != None:
                design = self.design
            else:
                design = [16]
        if isinstance(design, type(int())):
            design = [design]

        if thetas != None:
            design = []
            for theta in thetas[:-1]:
                design.append(theta.shape[1])
        self.design = design

        if gamma != None:
            self.gamma = gamma

        layers = []
        nl = len(design) + 1
        for idx in range(nl):
            if thetas != None:
                theta = thetas[idx]
            else:
                theta = np.array([])

# add bias to all inputs
            if idx == 0:
                lin = nfeatures + 1 #add bias
            else: 
                lin = design[idx - 1] + 1

# add bias to outputs for internal/hidden layers
            if idx == nl - 1:
                lout = ntargets
            else:
                lout = design[idx]
            if shiftlayer is not None:
                if idx == shiftlayer:
                    layers.append(layer(lin, lout, theta, shiftlayer=True))
                else:
                    layers.append(layer(lin, lout, theta, shiftlayer=False))
            else:
                layers.append(layer(lin, lout, theta, shiftlayer=False))
                
        if verbose or self.verbose:
            txt = "Created (network,  gamma) = (%s-->" % (nfeatures)
            for idx in design:
                txt += "%s-->" % idx
            txt += "%s,  %s) " % (ntargets, self.gamma)
            print(txt)
        self.design = design
        self.ntargets = ntargets
        self.layers = layers
        self.nlayers = len(layers)


    def unflatten_thetas(self, thetas):
        """
        in order to use scipy.fmin functions we 
        need to make the Theta dependance explicit.
        
        This routine takes a flattened array 'thetas' of
        all the internal layer 'thetas', then assigns them
        onto their respective layers
        (ordered from earliest to latest layer)

        """
        bi = 0
        for lv in self.layers:
            shape = lv.theta.shape
            ei = bi + shape[0] * shape[1]
            lv.theta = thetas[bi:ei].reshape(shape)
            bi = ei

    def flatten_thetas(self):
        """
        in order to use scipy.fmin functions we
        need to make the layer's Theta dependencies explicit.

        this routine returns a giant array of all the flattened
        internal theta's, from earliest to latest layers

        """
        z = np.array([])
        for lv in self.layers:
            z = np.hstack([z, lv.theta.flatten()])
        return z

    def costFunctionU(self, X, y, gamma=None):
        """
        routine which calls costFunction, but
        unwraps the internal parameters (theta's)
        for you

        """
        thetas = self.flatten_thetas()
        return self.costFunction(thetas, X, y, gamma)

    def costFunction(self, thetas, X, y, gamma=None, verbose=None):
        """
        determine the cost function for this neural network
        given the training data X, classifcations y, 
        and learning rate (regularization) lambda

        Arguments:
        X : num_trials x ninputs, ninputs not including bias term
        return cost
        y : classification label for the num_trials
        gamma : regularization parameter, 
               default = None = self.gamma
        verbose: printing out training information (cost, #iterations)

        """
        global _niter
        if isinstance(X, type([])):
            X = np.array(X)
        
        if gamma == None:
            gamma = self.gamma

# testing: check the theta's are changing while we train: yes!
#        print "CF",thetas[0:2], thetas[-5:-2]

# update the layer's theta's
        self.unflatten_thetas(thetas)
        
        # number of trials
        if X.ndim == 2:
            N = X.shape[0] 
        else:
            N = 1

        # propagate the input through the entire network
        z, h = self.forward_propagate(X)
        yy = labels2vectors(y, self.ntargets)
 
        J = 0.
        J = (-np.log(h) * yy.transpose() - np.log(1-h)*(1-yy.transpose())).sum()
        J = J/N

# regularize (ignoring bias):
        reg = 0.
        for l in self.layers:
            reg +=  (l.theta[1:, :]**2).sum()
        J = J + gamma*reg/(2*N)
        
        if verbose or self.verbose:
            if _niter % 25 == 0:
                sys.stdout.write("\r\t(fit %s) NN.fit iter %s, Cost %12.7f "
                                 % (self.nfit,_niter, J))
                sys.stdout.flush()
        _niter += 1
        return J

    def forward_propagate(self, z, nl=100):
        """
        given an array of samples [nsamples, nproperties]
        propagate the sample through the neural network,
        Returns:
        z, a : the ouputs and activations (z(nl),acts(nl)) at layer nl
        
        Args:
        X = [nsamples, nproperties] (no bias)
        nl = number of layers to propagte through
             defaults to end (well, one hundred layers!)
        """
        if isinstance(z, type([])):
            z = np.array(z)
        # number of trials
        if z.ndim == 2:
            N = z.shape[0] 
            # add bias
            a = np.hstack([np.ones((N, 1)), z])
        else:
            N = 1
            # add bias
            a = np.hstack([np.ones(N), z])

        final_layer = len(self.layers) - 1
        for li, lv in enumerate(self.layers[0:nl]):
            z = np.dot(a, lv.theta)
            # add bias to input of each internal layer
            if li != final_layer:
                if N == 1 and z.ndim == 1:
                    a = np.hstack([np.ones(N), sigmoid(z)])
                else:
                    if _fort_opt:
                        a = np.hstack([np.ones((N, 1)), sigmoid2d(z)])
                    else:
                        a = np.hstack([np.ones((N, 1)), sigmoid(z)])
            else:
                if N == 1:
                    a = sigmoid(z)
                else:
                    if _fort_opt:
                        a = sigmoid2d(z)
                    else:
                        a = sigmoid(z)
        return z, a

    def gradientU(self, X, y, gamma=None, shiftlayer=None, verbose=None):
        """
        Convenience function.
        routine which calls gradient, but
        unwraps the internal parameters (theta's)
        for you.

        needed for scipy.fmin functions

        """
        thetas = self.flatten_thetas()
        return self.gradient(thetas, X, y, gamma, verbose=verbose)
        
    def gradient(self, thetas, X, y, gamma=None, verbose=None):
        """
        compute the gradient at each layer of the neural network
        
        Args:
        X = [nsamples, ninputs]
        y = [nsamples] #the training classifications
        gamma : regularization parameter
               default = None = self.gamma
        shiftlayer : None, or the layer which 

        returns the gradient for the parameters of the neural network
        (the theta's) unrolled into one large vector, ordered from
        the first layer to latest.

        """
        if isinstance(X, type([])):
            X = np.array(X)

        if gamma == None:
            gamma = self.gamma

        N = X.shape[0]
        nl = len(self.layers)
# create our grad_theta arrays (init to zero):
        grads = {}
        for li, lv in enumerate(self.layers):
            grads[li] = np.zeros_like(lv.theta)

# vectorize sample-loop
        if 1:
            for li in range(nl, 0, -1):
                z, a = self.forward_propagate(X,li)

                if li == nl:
                    ay = labels2vectors(y, self.ntargets).transpose()
                    delta = (a - ay)
                else:
                    theta = self.layers[li].theta
                    aprime = np.hstack([np.ones((N,1)), sigmoidGradient(z)]) #add in bias
#use fortran matmult if arrays are large
                    if _fort_opt and deltan.size * theta.size > 100000:
                       tmp = matmult(deltan,theta.transpose())
                    else:
                       tmp = np.dot(deltan,theta.transpose())#nsamples x neurons(li)
                    delta = tmp*aprime
                    
#find contribution to grad
                idx = li - 1
                z, a = self.forward_propagate(X,idx)
                if idx in grads:
                    if li == nl:
                        grads[idx] = np.dot(a.transpose(), delta)/N
                    else:
                        #strip off bias
                        grads[idx] = np.dot(a.transpose(), delta[:,1:])/N

#if this is a "shift-invar" layer, find the average grad 
                if self.shiftlayer is not None:
                    if li == self.shiftlayer:
                        shape = self.layers[li].theta.shape
                        grads = grads.reshape(shape)
                        l1 = grads[:,0]
                        dshift = N//m
                        for i in range(m-1):
                            shift = -dshift * (i+1) #undo the previous shifts
                            self.theta[:,i+1] = np.roll(l1, shift)
                        grad_avg = self.theta.mean(axis=0)
                        grads = np.array([grad_avg for i in range(shape[1])]).flatten()

#keep this delta for the next (earlier) layer
                if li == nl:
                    deltan = delta
                else:
                    deltan = delta[:,1:]


#now regularize the grads (bias doesn't get get regularized):
        for li, lv in enumerate(self.layers):
            theta = lv.theta
            grads[li][:, 1:] = grads[li][:, 1:] + gamma/N*theta[:, 1:]
            
#finally, flatten the gradients
        z = np.array([])
        for k in sorted(grads):
            v = grads[k]
            z = np.hstack([z, v.flatten()])
        return z

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training set.

        y : array-like, shape = [n_samples]
            Labels for X.

        Returns
        -------
        z : float

        """
        return np.mean(self.predict(X) == y)

    def fit(self, X, y, design=None, gamma=None,
            gtol=1e-05, epsilon=1.4901161193847656e-08, maxiter=None,
            raninit=True, info=False, verbose=None, fit_type=None):
        """
        Train the data.
        minimize the cost function (wrt the Theta's)
        (using the conjugate gradient algorithm from scipy)
        This updates the NN.layers.theta's, so one can
        later "predict" other samples. 
        ** The number of input features is determined from X.shape[1],
        and the number of targets from np.unique(y), so make sure
        'y' spans your target space.

        Args:
        X : the training samples [nsamples x nproperties]
        y : the sample labels [nsamples], each entry in range 0<=y<nclass
        design : list of number of neurons in each layer. 
             Default = None, uses self.create_layers default=[16]
             Eg. design=[12,3] is a neural network with 2 layers of 12, then 3 neurons
                 (we add bias later)
        gamma : regularization parameter
               default = None = self.gammas
        verbose : print out layer-creation information
        *for scipy.optimize.fmin_cg:
        gtol
        epsilon
        maxiter
        info : T/F, return the information from fmin_cg (Default False)
        
        raninit : T/F randomly initialize the theta's [default = True]
                (only use 'False' to continue training, not for new NN's) 
        """
        global _niter
        _niter = 0

        if gamma == None:
            gamma = self.gamma
        if verbose or self.verbose:
            verbose = True
        if fit_type == None:
            fit_type = self.fit_type
        if maxiter == None:
            maxiter = self.maxiter
            
        # train all layers at same time
        if fit_type == 'all':
    #update the NN layers (if necessary)
            if raninit:
                self.create_layers(X.shape[1], 
                                   np.unique(y).size, 
                                   design=design,
                                   gamma=gamma,
                                   verbose=verbose,
                                   shiftlayer=self.shiftlayer)
                for lv in self.layers:
                    lv.randomize()

            thetas = self.flatten_thetas()
            xopt = fmin_cg(f=self.costFunction,
                           x0=thetas,
                           fprime=self.gradient,
                           args=(X, y, gamma, verbose), #extra args to costFunction 
                           maxiter=maxiter,
                           epsilon=epsilon,
                           gtol=gtol,
                           disp=0,
                           )

        # build up the NN, training each layer one at a time
        elif fit_type == 'single':
            if design == None:
                design = self.design
            if self.nin == None:
                self.nin = X.shape[1]
                self.nout = np.unique(y).size

            # train the individual layers
            thetas = []
            for lyr in range(len(design)):
                if verbose:
                    print "\nTraining layer %s" % (lyr+1)
                nn = NeuralNetwork(gamma = gamma,
                                   design = design[0:lyr+1],
                                   fit_type='all'
                                   )
                nn.create_layers(self.nin, 
                                 self.nout,
                                 design=design[0:lyr+1],
                                 gamma=gamma,
                                 verbose=verbose,
                                 shiftlayer=self.shiftlayer)
                for lyri, theta in enumerate(thetas):
                    nn.layers[lyri].theta = theta
                nn.fit(X, y,
                       gtol=gtol, epsilon=epsilon, maxiter=maxiter,
                       raninit=False, info=info, verbose=verbose)
                thetas =[nn.layers[i].theta for i in range(lyr+1)] 
#.append(nn.layers[lyr].theta)
            #end design loop

# (diagnostics)
#                if verbose:
#                    nn.plot_firstlayer()

            # transfer the Theta's to our NN
            self.create_layers(X.shape[1], 
                               np.unique(y).size, 
                               design=design,
                               gamma=gamma,
                               verbose=verbose,
                               shiftlayer=self.shiftlayer)
#            for lyri, theta in enumerate(thetas):
#                self.layers[lyri].theta = theta
            for lyri, lyr in enumerate(nn.layers):
                self.layers[lyri].theta = lyr.theta

#            print "N",len(self.layers),self.layers[-1].theta[0:3,0:3]
#            print "O",len(thetas),thetas[-1][0:3,0:3]
           
            self.fit(X, y,
                     gtol=gtol, epsilon=epsilon, maxiter=maxiter,
                     raninit=False, info=info, verbose=verbose,
                     fit_type='all') 
                                   
        self.nfit += 1

        if info:
            print("\n")
            return xopt
        
    def predict(self, X):
        """
        Given a list of samples, predict their class.
        One should run nn.fit first to train the neural network.

        Args:
        X = [nsamples x nproperties]

        returns:
        y = [nsamples]
        
        """
        if isinstance(X, type([])):
            X = np.array(X)
        if len(X.shape) == 2:
            N = X.shape[0] 
        else:
            N = 1

        z, h  = self.forward_propagate(X)
        #find most-active label
        if N == 1:
            cls = np.array([h.argmax()])
        else:
            cls = h.argmax(axis=1)
        return cls
    
    def predict_proba(self, X):
        """
        Compute the likehoods each possible outcomes of samples in T.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        X : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in
            the model, where classes are ordered by arithmetical
            order.

        """
        if isinstance(X, type([])):
            X = np.array(X)
        if len(X.shape) == 2:
            N = X.shape[0] 
        else:
            N = 1

        z, h  = self.forward_propagate(X)
        norm = h.sum(axis=1)
        for ni, nv in enumerate(norm):
            h[ni] = h[ni] / nv
        return h

    def score_weiwei(self, X, y, verbose=None):
        """
        Returns the mean accuracy on the given test data and labels
    
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Training set.
        
        y : array-like, shape = [n_samples]
        Labels for X.
        
        Returns
        -------
        z : float

        """
        pred_cls = {}
        true_cls = {}
        for cls in range(self.ntargets):
            pred_cls[cls] = set([])
            true_cls[cls] = set([])
            
        for i, s in enumerate(X):
            predict = self.predict(s)[0]
            true_cls[y[i]].add(i)
            pred_cls[y[i]].add(i)

        tot_acc = 0.
        for k in range(self.ntargets):
            hit = pred_cls[k] & true_cls[k]
            miss = pred_cls[k] - true_cls[k]
            falsepos = true_cls[k] - pred_cls[k] 
            precision = np.divide(float(len(hit)), len(pred_cls[k]))
            recall = np.divide(float(len(hit)), len(true_cls[k]))
            accuracy = (np.divide(float(len(hit)), len(true_cls[k])) * 100)
            tot_acc += accuracy
            if verbose or self.verbose:
                print "\nClass %s:" % k
                print 'accuracy: ', '%.0f%%' % (np.divide(float(len(hit)),len(true_cls[k])) * 100)
                print 'miss: ', '%.0f%%' % (np.divide(float(len(miss)),len(true_cls[k])) * 100)
                print 'false positives: ', '%.0f%%' % (np.divide(float(len(falsepos)),len(pred_cls[k]))* 100)
                print 'precision: ', '%.0f%%' % (precision* 100)
                print 'recall: ', '%.0f%%' % (recall* 100)

        z = tot_acc / self.ntargets
        return z

    def learning_curve(self, X, y,
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
        if Xval == None:
            X, y, Xval, yval = split_data(X, y, pct=pct)

        if gamma == None:
            gamma = self.gamma

        m = X.shape[0]
#need at least one training item...
        stepsize = max(m/25,1)
        ntrials = range(1,m,stepsize)
        mm = len(ntrials)
        t_error = np.zeros(mm)
        v_error = np.zeros(mm)
        for i, v in enumerate(ntrials):
            #fit with regularization
            self.fit(X[0:v+1], y[0:v+1], gamma=gamma, maxiter=50, raninit=True)
            
            # but compute error without regularization
            t_error[i] = self.costFunctionU(X[0:v+1], y[0:v+1], gamma=0.)
            # use entire x-val set
            v_error[i] = self.costFunctionU(Xval, yval, gamma=0.)
            
        if plot:
            plt.plot(ntrials, t_error, 'r+', label='training')
            plt.plot(ntrials, v_error, 'bx', label='x-val')
            plt.xlabel('training set size')
            plt.ylabel('error [J(gamma=0)]')
            plt.legend()
            plt.show()

        return t_error, v_error, ntrials

    def validation_curve(self, X, y,
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
              Again, we train with regularization, but the erorr
              is calculated without

        returns:
        train_error(gamma), cross_val_error(gamma), gamma, best_gamma

        """
        if Xval == None:
            X, y, Xval, yval = split_data(X, y, pct)
        
        if gammas == None:
            gammas = [0., 0.0001, 0.0005, 0.001, 0.05, 0.1, .5, 1., 1.5, 15.]
        
        train_error = np.zeros(len(gammas))
        xval_error = np.zeros(len(gammas))
        for gi, gv in enumerate(gammas):
#train with reg.
            self.fit(X, y, gamma=gv, maxiter=40, raninit=True)
           
#evaluate error without reg. 
            train_error[gi] = self.costFunctionU(X, y, gamma=0.)
            xval_error[gi] = self.costFunctionU(Xval, yval, gamma=0.)

        if plot:
            plt.plot(gammas, train_error, label='Train')
            plt.plot(gammas, xval_error, label='Cross Validation')
            plt.xlabel('gamma')
            plt.ylabel('Error [costFunction]')
            plt.legend()
            plt.show()

        return train_error, xval_error, gammas, gammas[xval_error.argmin()]

    def numericalGradients(self, X, y):
        """
        numerically estimate the gradients using finite differences
        (used to compare to 'gradient' routine)

        loop over layers, perturbing each theta-parameter one at a time


        * useful for testing gradient routine *
        """
        from copy import deepcopy

        thetas = self.flatten_thetas()
        origthetas = deepcopy(thetas)
        numgrad = np.zeros(thetas.size)
        perturb = np.zeros(thetas.size)

        delta = 1.e-4

        for p in range(numgrad.size):
            #set the perturbation vector
            perturb[p] = delta
            loss1 = self.costFunction(thetas - perturb, X, y)
            loss2 = self.costFunction(thetas + perturb, X, y)
            #calculat the numerical gradient
            numgrad[p] = (loss2 - loss1) / (2*delta)
            #reset the perturbation
            self.unflatten_thetas(origthetas)
            perturb[p] = 0
            
## OLD
# loop over layers, neurons
        idx = 0
        if 0: 
            for lv in self.layers:
                theta_orig = deepcopy(lv.theta)

# perturb each neuron and calc. grad at that neuron
                for pi in range(theta_orig.size): #strip bias
                    perturb = np.zeros(theta_orig.size)
                    perturb[pi] = delta
                    perturb = perturb.reshape(theta_orig.shape)
                    lv.theta = theta_orig + perturb
                    loss1 = self.costFunctionU(X, y)
                    lv.theta = theta_orig - perturb
                    loss2 = self.costFunctionU(X, y)
                    numgrad[idx] = (loss2 - loss1) / (2*delta)
                    idx += 1
                lv.theta = theta_orig

        return numgrad

    def pickle_me(self, filename=''):
        """
        dump the important parts of the classifier to a pickled file.
        (the theta's and gamma=learning rate)
        We name the file based on number of layers,
        and shape of layers
        
        args:
        filename : base filename, append layer info to this

        """
        for li, lv in enumerate(self.layers):
            if filename:
                filename += '_l%sx%s' % lv.theta.shape
            else:
                filename += 'l%sx%s' % lv.theta.shape
        filename += '.pkl'
#create our pickle
        d = {}
        d['nlayers'] = len(self.layers)
        for li, lv in enumerate(self.layers):
            d[li] = lv.theta
        d['gamma'] = self.gamma

        print "pickling classifier to %s" % filename
        cPickle.dump(d, open(filename, 'w'))
        
    def write_thetas(self, basename='layer_'):
        """
        Write the theta's to a file in form
        basename_%s.npy % layer_number

        args
        basename : the base filename to write the data to
       
        """
        
        for li, lv in enumerate(self.layers):
            theta = lv.theta
            name = "%s_%s.npy" % (basename, li)
            print "Saving layer %s to %s " % (li, name)
            np.save(name, theta)

    def plot_firstlayer(self, imgdim=None,Nsamples=None):
        """
        plot the first layer of neurons
        
        Args:
        imgdim = array [n, m]  of input image dimensions
        Nsamples: randomly choose Nsamples-by-Nsamples of neurons
                 and plot them. 
                default = None = plot all neurons in first layer

        """
        theta = self.layers[0].theta[1:,:] #strip off bias
        nimg, nneurons = theta.shape

        if imgdim == None:
            nx = int(np.sqrt(nimg))
            ny = nx
            if nx*ny < nimg:
                nx += 1
        else:
            nx, ny = imgdim

        if nx*ny != nimg:
            #Assume this is a 1-d thingy
#            print "image dimensions (%s,%s) don't match actual %s" % (nx,ny,nimg)
            nx = nimg
            ny = 1
#           return

        if Nsamples == None:
#plot all of them
            Nsamples = int(np.sqrt(nneurons))
            if Nsamples*Nsamples < nneurons:
                Nsamples += 1
            samples = theta
            print "showing all neurons"
        else:
            if nneurons < Nsamples*Nsamples:
            #not enough neurons to fill this matrix completely, so use them all
                Nsamples = int(np.sqrt(nneurons))
                if Nsamples*Nsamples < nneurons:
                    Nsamples += 1
                samples = theta
            else:
            #randomly choose a subset
                n = np.random.uniform(0, nneurons, Nsamples**2).astype(np.int)
                samples = theta[:,n]
        
#add buffer between neurons:
        if ny != 1:
            buf = 3
            data = np.zeros((Nsamples*(nx+buf), Nsamples*(ny+buf)))
            for xi, xv in enumerate(samples.transpose()):
                col = xi % Nsamples
                row = xi // Nsamples
    #            print xi,data.shape,row,col,xv.shape
                data[row*(nx+buf):(row+1)*nx+row*buf, col*(ny+buf):(col+1)*ny+col*buf] = xv.reshape(nx,ny)

            plt.imshow(data, cmap=plt.cm.gray)
        else:
            for xi, xv in enumerate(samples.transpose()):
                plt.plot(xv.reshape(nx,ny))
        plt.title('First layer of Neural Network')
        plt.show()
        
############# end class NeuralNetwork ***************


def checkGradients(nin=3, nout=3, ninternal=np.array([5]),
                   Nsamples=5, gamma=0.):
    """
    Create a small neural network to check
    backpropagatoin gradients

    this routine compares analytical gradients to the numerical gradients
    (as computed by compute_numericalGradients)

    returns
    numerical_gradient, gradient
    
    """
    print "Creating small NN to compare numerical approx to gradient"
    print "with actual gradient. Arrays should be identical"

#create the neural network and some fake data
    X = np.random.random((Nsamples, nin))
#labels from 0 <= y < nlabels = nout
    y =  np.array(np.random.uniform(0, nout, Nsamples), dtype=int)

# neural network, delta=0 and thetas=[] --> theta is randomly inited.
    nn = create_NN(nin, nout, ninternal=ninternal, thetas=np.array([]), delta=0)

    numgrad = nn.numericalGradients(X, y)
    grad = nn.gradientU(X, y, gamma)
    return numgrad, grad
        
class handwritingpix(object):
    """
    read in the machine learning data files, which we 
    use to test our python implementation of the neural network

    we assume they are 20x20 pixels 
    initializes with a filename, assumes matlab format
    though can have text=True for text format
    then requires the trained values 'y'
    """
    def __init__(self, samples, thetas):

        self.fname = samples

        data = io.loadmat(samples)
        self.data = data['X']
        self.y = data['y']
# make sure the labels are in range 0<=y<nclass 
# so we can easily index our row vector
        self.y = self.y - self.y.min()
        thetas = io.loadmat(thetas)
        self.theta1 = thetas['Theta1'].transpose()
        self.theta2 = thetas['Theta2'].transpose()

#assume square image
        self.N = self.data.shape[1]
        self.Nsamples = self.data.shape[0]
        self.nx = np.sqrt(self.N)
        self.ny = np.sqrt(self.N)
        self.thetas = [self.theta1, self.theta2]

    def plot_samples(self, Nsamples=10):
        """
        randomly pick Nsamples**2 and plot them

        """
        nx = self.nx
        ny = self.ny

        n = np.random.uniform(0, self.Nsamples, Nsamples**2).astype(np.int)
        samples = self.data[n, :]
        
        data = np.zeros((Nsamples*ny, Nsamples*nx))
        print n
        for xi, xv in enumerate(samples):
            col = xi % Nsamples
            row = xi // Nsamples
#            print xi,data.shape,row,col,xv.shape
            data[row*ny:(row+1)*ny, col*nx:(col+1)*nx] = xv.reshape(20, 20)
            
        plt.imshow(data, cmap=plt.cm.gray)
        plt.show()
        


#### Utility Functions ####
def feature_curve(feature,
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
    target = originaldata['target']
    classdict = {0:[4,5], 1:[6,7]}
    for k, v in classdict.iteritems():
        for val in v:
            target[target == val] = k

    if bounds == None:
        vals = mgrid[8:32:1j*Npts]
    else:
        vals = mgrid[bounds[0]:bounds[1]:1j*Npts]

    kws = {'phasebins':0}
    train_score = np.zeros_like(vals)
    test_score = np.zeros_like(vals)
    for i, val in enumerate(vals):
        kws[feature] = int(val)
        data = [pf.getdata(**kws) for pf in pfds]
        train_data, train_target, test_data, test_target = split_data(data,target, pct=pct)
        classifier = create_NN(train_data,train_target,
                               ninternal=[9],
                               gamma=0.00025)
        print "\nfeature idx, size, data_shape: %s, %s, %s" %\
            (i, int(val),train_data.shape)

        classifier.fit(train_data,train_target,maxiter=2222,raninit=True)
#record erro
        train_score[i] = 1-classifier.score(train_data, train_target)
        test_score[i] = 1-classifier.score(test_data, test_target)
    if plot:
        plt.plot(vals, train_score, 'r+', label='training')
        plt.plot(vals, test_score, 'bx', label='x-val')
        plt.xlabel(feature)
        plt.ylabel('error')
        plt.legend()
        plt.show()
    return train_score, test_score, vals, vals[test_score.argmax()]


def labels2vectors(y, Nclass=1):
    """
    given a vector of [nsamples] where the i'th entry is label/classification
    for the i'th sample, return an array [nlabels,nsamples],
    projecting each sample 'y' into the appropriate row

    args:
    y : [nsamples] 
    Nclass: the number of classifications,
            defaults to number of unique items in y

    **assumes classifications are in range 0 <= y < Nclass
    """
    if isinstance(y, np.array([]).__class__):
        pass
    else:
        y = np.array([y], dtype=np.int)

# number of samples
    N = y.size

# determine number of classes
    if Nclass:
        nclass = Nclass
    else:
        nclass = len(np.unique(y))


# map labels onto column vectors
    if N == 1:
        yy = np.zeros(nclass, dtype=np.uint8)
    else:
        yy = np.zeros((nclass, N), dtype=np.uint8)

    for yi, yv in enumerate(y):
        if N == 1:
            yy[yv] = 1
        else:
            yy[yv, yi] = 1
    return yy

def sigmoid(z):
    """
    compute element-wise the sigmoid of input array

    """
    return 1./(1.0 + np.exp(-z))

def sigmoidGradient(z):
    """
    compute element-wise the sigmoid-Gradient of input array

    """
    return sigmoid(z) * (1-sigmoid(z))
        
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



if __name__ == '__main__':
    main()
