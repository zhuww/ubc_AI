"""
Aaron Berndsen:
A Conformal Neural Network using Theano for computation and structure, 
but built to obey sklearn's basic 'fit' 'predict' functionality

*code largely motivated from deeplearning.net examples
and Graham Taylor's "Vanilla RNN" (https://github.com/gwtaylor/theano-rnn/blob/master/rnn.py)

You'll require theano and libblas-dev

tips/tricks/notes:
* if training set is large (>O(100)) and redundant, use stochastic gradient descent (batch_size=1), otherwise use conjugate descent (batch_size > 1)
*  
"""
import cPickle as pickle
import logging
import numpy as np
from collections import OrderedDict

from sklearn.base import BaseEstimator
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import logging
_logger = logging.getLogger("theano.gof.compilelock")
_logger.setLevel(logging.WARN)
logger = logging.getLogger(__name__)

mode = theano.Mode(linker='cvm')
#mode = 'DEBUG_MODE'

class CNN(object):
    """
    Conformal Neural Network, 
    backend by Theano, but compliant with sklearn interface.

    This class holds the actual layers, while MetaCNN does
    the fit/predict routines. You should init with MetaCNN.
    
    There are three layers:
    layer0 : a convolutional filter making filters[0] shifted copies,
             then downsampled by max pooling in grids of poolsize[0]
             (N, 1, nx, ny)
             --> (N, nkerns[0], nx1, ny1)  (nx1 = nx - filters[0][0] + 1)
                                  (ny1 = ny - filters[0][1] + 1)
             --> (N, nkerns[0], nx1/poolsize[0][1], ny1/poolsize[0][1])
    layer1 : a convolutional filter making filters[1] shifted copies,
             then downsampled by max pooling in grids of poolsize[1]
             (N, nkerns[0], nx1/2, ny1/2)
             --> (N, nkerns[1], nx2, ny2) (nx2 = nx1 - filters[1][0] + 1)
             --> (N, nkerns[1], nx3, ny3) (nx3 = nx2/poolsize[1][0], ny3=ny2/poolsize[1][1])
    layer2 : hidden layer of nkerns[1]*nx3*ny3 input features and n_hidden hidden neurons
    layer3 : final LR layer with n_hidden neural inputs and n_out outputs/classes
             

    """
    def __init__(self, input, n_in=1, n_out=0, activation=T.tanh,
                 nkerns=[20,50],
                 filters=[15,9],
                 poolsize=[(3,3),(2,2)],
                 n_hidden=500,
                 output_type='softmax', batch_size=25,
                 use_symbolic_softmax=False):

        """
        n_in : width (or length) of input image (assumed square)
        n_out : number of class labels
        
        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer
        
        :type filters: list of ints, or 2-tuples
        :param filters: width of convolution. 
                        if 2-tuples, filter size can be different in x and y direction

        :type poolsize: list of 2-tuples
        :param poolsize: maxpooling in convolution layer (index-0),
                         and direction x or y (index-1)

        :type n_hidden: int
        :param n_hidden: number of hidden neurons
        
        :type output_type: string
        :param output_type: type of decision 'softmax', 'binary', 'real'
        
        :type batch_size: int
        :param batch_size: number of samples in each training batch. Default 200.
        """    
        self.activation = activation
        self.output_type = output_type

        #shape of input images
        nx, ny = n_in, n_in

        if use_symbolic_softmax:
            def symbolic_softmax(x):
                e = T.exp(x)
                return e / T.sum(e, axis=1).dimshuffle(0, 'x')
            self.softmax = symbolic_softmax
        else:
            self.softmax = T.nnet.softmax

        # Reshape matrix of rasterized images of shape (batch_size, nx*ny)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        layer0_input = input.reshape((batch_size, 1, nx, ny))

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (nx-filx+1,ny-fily+1)
        # maxpooling reduces this further to (nx/poosize[0][0],ny/poolsize[0][1]) 
        # 4D output tensor is thus of shape (batch_size,nkerns[0],xx,yy)
        nim = filters[0]
        if isinstance(nim, int):
            fil1x = nim
            fil1y = nim
        else:
            fil1x = nim[0]
            fil1y = nim[1]
        rng = np.random.RandomState(23455)
        self.layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
                                  image_shape=(batch_size, 1, nx, ny),
                                  filter_shape=(nkerns[0], 1, fil1x, fil1y),
                                         poolsize=poolsize[0])
        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (nbin-nim+1,nbin-nim+1) = x
        # maxpooling reduces this further to (x/2,x/2) = y
        # 4D output tensor is thus of shape (nkerns[0],nkerns[1],y,y)
        poox = (nx - fil1x + 1)/poolsize[0][0]
        pooy = (ny - fil1y + 1)/poolsize[0][1]
        nconf = filters[1]
        if isinstance(nconf, int):
            fil2x = nconf
            fil2y = nconf
        else:
            fil2x = nconf[0]
            fil2y = nconf[1]
        self.layer1 = LeNetConvPoolLayer(rng, input=self.layer0.output,
                image_shape=(batch_size, nkerns[0], poox, pooy),
                filter_shape=(nkerns[1], nkerns[0], fil2x, fil2y),
                                         poolsize=poolsize[1])

        # the TanhLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (20,32*4*4) = (20,512)
        layer2_input = self.layer1.output.flatten(2)

       # construct a fully-connected sigmoidal layer
        poo2x = (poox - fil2x + 1)/poolsize[1][0]
        poo2y = (pooy - fil2y + 1)/poolsize[1][1]
        self.layer2 = HiddenLayer(rng, input=layer2_input,
                                  n_in=nkerns[1]*poo2x*poo2y,
                                  n_out=n_hidden, activation=T.tanh)

        # classify the values of the fully-connected sigmoidal layer
        self.layer3 = LogisticRegression(input=self.layer2.output,
                                         n_in=n_hidden, n_out=n_out)

        # CNN regularization
        self.L1 = self.layer3.L1
        self.L2_sqr = self.layer3.L2_sqr
        
        # create a list of all model parameters to be fit by gradient descent
        self.params = self.layer3.params + self.layer2.params\
            + self.layer1.params + self.layer0.params

        self.y_pred = self.layer3.y_pred
        self.p_y_given_x = self.layer3.p_y_given_x

        if self.output_type == 'real':
            self.loss = lambda y: self.mse(y)
        elif self.output_type == 'binary':
            self.loss = lambda y: self.nll_binary(y)
        elif self.output_type == 'softmax':
            # push through softmax, computing vector of class-membership
            # probabilities in symbolic form
            self.loss = lambda y: self.nll_multiclass(y)
        else:
            raise NotImplementedError

    def mse(self, y):
        # error between output and target
        return T.mean((self.y_pred - y) ** 2)

    def nll_binary(self, y):
        # negative log likelihood based on binary cross entropy error
        return T.mean(T.nnet.binary_crossentropy(self.p_y_given_x, y))

    #same as negative-log-likelikhood
    def nll_multiclass(self, y):
        # negative log likelihood based on multiclass cross entropy error
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of time steps (call it T) in the sequence
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the sequence
        over the total number of examples in the sequence ; zero one
        loss over the size of the sequence

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        # check if y has same dimension of y_pred
        if y.ndim != self.y_out.ndim:
            raise TypeError('y should have the same shape as self.y_out',
                ('y', y.type, 'y_pred', self.y_pred.type))

        if self.output_type in ('binary', 'softmax'):
            # check if y is of the correct datatype
            if y.dtype.startswith('int'):
                # the T.neq operator returns a vector of 0s and 1s, where 1
                # represents a mistake in prediction
                return T.mean(T.neq(self.y_pred, y))
            else:
                raise NotImplementedError()


class MetaCNN(BaseEstimator):
    """
    the actual CNN is not init-ed until .fit is called.
    We determine the image input size (assumed square images) and
    the number of outputs in .fit from the training data

    """
    def __init__(self, learning_rate=0.05,
                 n_epochs=60, batch_size=25, activation='tanh', 
                 nkerns=[20,45],
                 n_hidden=500,
                 filters=[15,7],
                 poolsize=[(3,3),(2,2)],
                 output_type='softmax',
                 L1_reg=0.00, L2_reg=0.00,
                 use_symbolic_softmax=False,
                 ### Note, n_in and n_out are actually set in 
                 ### .fit, they are here to help cPickle
                 n_in=50, n_out=2):
        self.learning_rate = float(learning_rate)
        self.nkerns = nkerns
        self.n_hidden = n_hidden
        self.filters = filters
        self.poolsize = poolsize
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.L1_reg = float(L1_reg)
        self.L2_reg = float(L2_reg)
        self.activation = activation
        self.output_type = output_type
        self.use_symbolic_softmax = use_symbolic_softmax
        self.n_in = n_in
        self.n_out = n_out

    def ready(self):
        """
        this routine is called from "fit" since we determine the
        image size (assumed square) and output labels from the training data.

        """
        #input
        self.x = T.matrix('x')
        #output (a label)
        self.y = T.ivector('y')
        
        if self.activation == 'tanh':
            activation = T.tanh
        elif self.activation == 'sigmoid':
            activation = T.nnet.sigmoid
        elif self.activation == 'relu':
            activation = lambda x: x * (x > 0)
        elif self.activation == 'cappedrelu':
            activation = lambda x: T.minimum(x * (x > 0), 6)
        else:
            raise NotImplementedError
        
        self.cnn = CNN(input=self.x, n_in=self.n_in, 
                       n_out=self.n_out, activation=activation, 
                       nkerns=self.nkerns,
                       filters=self.filters,
                       n_hidden=self.n_hidden,
                       poolsize=self.poolsize,
                       output_type=self.output_type,
                       batch_size=self.batch_size,
                       use_symbolic_softmax=self.use_symbolic_softmax)
        
        #self.cnn.predict expects batch_size number of inputs. 
        #we wrap those functions and pad as necessary in 'def predict' and 'def predict_proba'
        self.predict_wrap = theano.function(inputs=[self.x],
                                            outputs=self.cnn.y_pred,
                                            mode=mode)
        self.predict_proba_wrap = theano.function(inputs=[self.x],
                                                  outputs=self.cnn.p_y_given_x,
                                                  mode=mode)


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


    def fit(self, X_train, Y_train, X_test=None, Y_test=None,
            validation_frequency=2, n_epochs=None):
        """ Fit model

        Pass in X_test, Y_test to compute test error and report during
        training.

        X_train : ndarray (T x n_in)
        Y_train : ndarray (T x n_out)

        validation_frequency : int
            in terms of number of sequences (or number of weight updates)
        n_epochs : None (used to override self.n_epochs from init.
        """
        #prepare the CNN 
        self.n_in = int(np.sqrt(X_train.shape[1]))
        self.n_out = len(np.unique(Y_train))
        self.ready()

        if X_test is not None:
            assert(Y_test is not None)
            interactive = True
            test_set_x, test_set_y = self.shared_dataset((X_test, Y_test))
        else:
            interactive = False

        train_set_x, train_set_y = self.shared_dataset((X_train, Y_train))

        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_train_batches /= self.batch_size

        if interactive:
            n_test_batches = test_set_x.get_value(borrow=True).shape[0]
            n_test_batches /= self.batch_size

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        logger.info('... building the model')

        index = T.lscalar('index')    # index to a [mini]batch

        cost = self.cnn.loss(self.y)\
            + self.L1_reg * self.cnn.L1\
            + self.L2_reg * self.cnn.L2_sqr

        compute_train_error = theano.function(inputs=[index, ],
                                              outputs=self.cnn.loss(self.y),
                                              givens={
                self.x: train_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                self.y: train_set_y[index * self.batch_size: (index + 1) * self.batch_size]},
                                              mode=mode)

        if interactive:
            compute_test_error = theano.function(inputs=[index, ],
                                                 outputs=self.cnn.loss(self.y),
                                                 givens={
                self.x: test_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                self.y: test_set_y[index * self.batch_size: (index + 1) * self.batch_size]},
                                                 mode=mode)

        # create a list of all model parameters to be fit by gradient descent
        self.params = self.cnn.params

        # create a list of gradients for all model parameters
        self.grads = T.grad(cost, self.params)
        
        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates dictionary by automatically looping over all
        # (params[i],grads[i]) pairs.
        self.updates = OrderedDict()
        for param_i, grad_i in zip(self.params, self.grads):
            self.updates[param_i] = param_i - self.learning_rate * grad_i

        train_model = theano.function([index], cost, updates=self.updates,
                                      givens={
                self.x: train_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                self.y: train_set_y[index * self.batch_size: (index + 1) * self.batch_size]}
                                      )

        ###############
        # TRAIN MODEL #
        ###############
        logger.info('... training')
        
        # early-stopping parameters
        patience = 1000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch
        best_test_loss = np.inf
        best_iter = 0
        epoch = 0
        done_looping = False 

        if n_epochs is None:
            n_epochs = self.n_epochs

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for idx in xrange(n_train_batches):

                iter = epoch * n_train_batches + idx

                cost_ij = train_model(idx)
                
                if iter % validation_frequency == 0:
                    # compute loss on training set
                    train_losses = [compute_train_error(i)
                                    for i in xrange(n_train_batches)]
                    this_train_loss = np.mean(train_losses)

                    if interactive:
                        test_losses = [compute_test_error(i)
                                        for i in xrange(n_test_batches)]
                        this_test_loss = np.mean(test_losses)
                        note = 'epoch %i, seq %i/%i, tr loss %f '\
                            'te loss %f lr: %f' % \
                            (epoch, idx + 1, n_train_batches,
                             this_train_loss, this_test_loss, self.learning_rate)
                        logger.info(note)
                        print note

                        if this_test_loss < best_test_loss:
                            #improve patience if loss improvement is good enough
                            if this_test_loss < best_test_loss *  \
                                    improvement_threshold:
                                patience = max(patience, iter * patience_increase)

                            # save best validation score and iteration number
                            best_test_loss = this_test_loss
                            best_iter = iter
                    else:
                        logger.info('epoch %i, seq %i/%i, train loss %f '
                                    'lr: %f' % \
                                    (epoch, idx + 1, n_train_batches, this_train_loss,
                                     self.learning_rate))
                if patience <= iter:
                    done_looping = True
                    break
        logger.info("Optimization complete")
        logger.info("Best xval score of %f %% obtained at iteration %i" %
                    (best_test_loss * 100., best_iter))


    def predict(self, data):
        """
        the CNN expects inputs with Nsamples = self.batch_size.
        In order to run 'predict' on an arbitrary number of samples we
        pad as necessary.

        """
        if isinstance(data, list):
            data = np.array(data)
        if data.ndim == 1:
            data = np.array([data])

        nsamples = data.shape[0]
        n_batches = nsamples//self.batch_size
        n_rem = nsamples%self.batch_size
        if n_batches > 0:
            preds = [list(self.predict_wrap(data[i*self.batch_size:(i+1)*self.batch_size]))\
                                           for i in range(n_batches)]
        else:
            preds = []
        if n_rem > 0:
            z = np.zeros((self.batch_size, self.n_in * self.n_in))
            z[0:n_rem] = data[n_batches*self.batch_size:n_batches*self.batch_size+n_rem]
            preds.append(self.predict_wrap(z)[0:n_rem])
        
        return np.hstack(preds).flatten()
    
    def predict_proba(self, data):
        """
        the CNN expects inputs with Nsamples = self.batch_size.
        In order to run 'predict_proba' on an arbitrary number of samples we
        pad as necessary.

        """
        if isinstance(data, list):
            data = np.array(data)
        if data.ndim == 1:
            data = np.array([data])

        nsamples = data.shape[0]
        n_batches = nsamples//self.batch_size
        n_rem = nsamples%self.batch_size
        if n_batches > 0:
            preds = [list(self.predict_proba_wrap(data[i*self.batch_size:(i+1)*self.batch_size]))\
                                           for i in range(n_batches)]
        else:
            preds = []
        if n_rem > 0:
            z = np.zeros((self.batch_size, self.n_in * self.n_in))
            z[0:n_rem] = data[n_batches*self.batch_size:n_batches*self.batch_size+n_rem]
            preds.append(self.predict_proba_wrap(z)[0:n_rem])
        
        return np.vstack(preds)
        

    def shared_dataset(self, data_xy):
        """ Load the dataset into shared variables """

        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX))

        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX))

        if self.output_type in ('binary', 'softmax'):
            return shared_x, T.cast(shared_y, 'int32')
        else:
            return shared_x, shared_y

    def __getstate__(self):
        """ Return state sequence."""
        
        #check if we're using ubc_AI.classifier wrapper, 
        #adding it's params to the state
        if hasattr(self, 'orig_class'):
            superparams = self.get_params()
            #now switch to orig. class (MetaCNN)
            oc = self.orig_class
            cc = self.__class__
            self.__class__ = oc
            params = self.get_params()
            for k, v in superparams.iteritems():
                params[k] = v
            self.__class__ = cc
        else:
            params = self.get_params()  #sklearn.BaseEstimator
        if hasattr(self, 'cnn'):
            weights = [p.get_value() for p in self.cnn.params]
        else:
            weights = []
        state = (params, weights)
        return state

    def _set_weights(self, weights):
        """ Set fittable parameters from weights sequence.

        Parameters must be in the order defined by self.params:
            W, W_in, W_out, h0, bh, by
        """
        i = iter(weights)
        if hasattr(self, 'cnn'):
            for param in self.cnn.params:
                param.set_value(i.next())

    def __setstate__(self, state):
        """ Set parameters from state sequence.

        Parameters must be in the order defined by self.params:
            W, W_in, W_out, h0, bh, by
        """
        params, weights = state
        #we may have several classes or superclasses
        for k in ['n_comp', 'use_pca', 'feature']:
            if k in params:
                self.set_params(**{k:params[k]})
                params.pop(k)

        #now switch to MetaCNN if necessary
        if hasattr(self,'orig_class'):
            cc = self.__class__
            oc = self.orig_class
            self.__class__ = oc
            self.set_params(**params)
            self.ready()
            if len(weights) > 0:
                self._set_weights(weights)
            self.__class__ = cc
        else:
            self.set_params(**params)
            self.ready()
            self._set_weights(weights)
            

    def save(self, fpath='.', fname=None):
        """ Save a pickled representation of Model state. """
        import datetime
        fpathstart, fpathext = os.path.splitext(fpath)
        if fpathext == '.pkl':
            # User supplied an absolute path to a pickle file
            fpath, fname = os.path.split(fpath)

        elif fname is None:
            # Generate filename based on date
            date_obj = datetime.datetime.now()
            date_str = date_obj.strftime('%Y-%m-%d-%H:%M:%S')
            class_name = self.__class__.__name__
            fname = '%s.%s.pkl' % (class_name, date_str)

        fabspath = os.path.join(fpath, fname)

        logger.info("Saving to %s ..." % fabspath)
        file = open(fabspath, 'wb')
        state = self.__getstate__()
        pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()

    def load(self, path):
        """ Load model parameters from path. """
        logger.info("Loading from %s ..." % path)
        file = open(path, 'rb')
        state = pickle.load(file)
        self.__setstate__(state)
        file.close()


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=np.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=np.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = 0
        self.L1 += abs(self.W.sum())

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = 0
        self.L2_sqr += (self.W ** 2).sum()

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)

        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(np.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

