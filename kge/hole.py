__author__ = 'nilesh'

from collections import OrderedDict
import keras
import theano as th
import theano.tensor as T

from keras import backend as K
from keras.optimizers import Adagrad, SGD
import keras
from keras.engine.topology import Layer
from keras.models import Sequential
from keras.layers import Dense, Activation, Input


class LanguageModel:
    def __init__(self, config):
        self.subject = Input(shape=(config['subject_len'],), dtype='int32', name='subject_base')
        self.subject_bad = Input(shape=(config['subject_len'],), dtype='int32', name='subject_bad_base')
        self.relation = Input(shape=(config['relation_len'],), dtype='int32', name='relation_base')
        self.object_good = Input(shape=(config['object_len'],), dtype='int32', name='object_good_base')
        self.object_bad = Input(shape=(config['object_len'],), dtype='int32', name='object_bad_base')

        self.config = config
        self.model_params = config.get('model_params', dict())
        self.similarity_params = config.get('similarity_params', dict())

        # initialize a bunch of variables that will be set later
        self._models = None
        self._similarities = None
        self._object = None
        self._subject = None
        self._qa_model = None
        self._qa_model_rt = None

        self.training_model = None
        self.training_model_rt = None
        self.prediction_model = None
        self.prediction_model_rt = None

    def get_object(self):
        if self._object is None:
            self._object = Input(shape=(self.config['object_len'],), dtype='int32', name='object')
        return self._object

    def get_subject(self):
        if self._subject is None:
            self._subject = Input(shape=(self.config['subject_len'],), dtype='int32', name='subject')
        return self._subject

    @abstractmethod
    def build(self):
        return

    def get_similarity(self):
        ''' Specify similarity in configuration under 'similarity_params' -> 'mode'
        If a parameter is needed for the model, specify it in 'similarity_params'
        Example configuration:
        config = {
            ... other parameters ...
            'similarity_params': {
                'mode': 'gesd',
                'gamma': 1,
                'c': 1,
            }
        }
        cosine: dot(a, b) / sqrt(dot(a, a) * dot(b, b))
        polynomial: (gamma * dot(a, b) + c) ^ d
        sigmoid: tanh(gamma * dot(a, b) + c)
        rbf: exp(-gamma * l2_norm(a-b) ^ 2)
        euclidean: 1 / (1 + l2_norm(a - b))
        exponential: exp(-gamma * l2_norm(a - b))
        gesd: euclidean * sigmoid
        aesd: (euclidean + sigmoid) / 2
        '''

        params = self.similarity_params
        similarity = params['mode']

        axis = lambda a: len(a._keras_shape) - 1
        dot = lambda a, b: K.batch_dot(a, b, axes=axis(a))
        l2_norm = lambda a, b: K.sqrt(K.sum((a - b) ** 2, axis=axis(a), keepdims=True))
        l1_norm = lambda a, b: K.sum(K.abs(a - b), axis=axis(a), keepdims=True)

        if similarity == 'cosine':
            return lambda x: dot(x[0], x[1]) / K.sqrt(dot(x[0], x[0]) * dot(x[1], x[1]))
        elif similarity == 'polynomial':
            return lambda x: (params['gamma'] * dot(x[0], x[1]) + params['c']) ** params['d']
        elif similarity == 'sigmoid':
            return lambda x: K.tanh(params['gamma'] * dot(x[0], x[1]) + params['c'])
        elif similarity == 'rbf':
            return lambda x: K.exp(-1 * params['gamma'] * l2_norm(x[0], x[1]) ** 2)
        elif similarity == 'euclidean':
            return lambda x: 1 / (1 + l2_norm(x[0], x[1]))
        elif similarity == 'l1':
            return lambda x: -l1_norm(x[0], x[1])
        elif similarity == 'exponential':
            return lambda x: K.exp(-1 * params['gamma'] * l2_norm(x[0], x[1]))
        elif similarity == 'gesd':
            euclidean = lambda x: 1 / (1 + l2_norm(x[0], x[1]))
            sigmoid = lambda x: 1 / (1 + K.exp(-1 * params['gamma'] * (dot(x[0], x[1]) + params['c'])))
            return lambda x: euclidean(x) * sigmoid(x)
        elif similarity == 'aesd':
            euclidean = lambda x: 0.5 / (1 + l2_norm(x[0], x[1]))
            sigmoid = lambda x: 0.5 / (1 + K.exp(-1 * params['gamma'] * (dot(x[0], x[1]) + params['c'])))
            return lambda x: euclidean(x) + sigmoid(x)
        else:
            raise Exception('Invalid similarity: {}'.format(similarity))

    def get_kge_model(self):
        if self._models is None:
            self._models = self.build()

        if self._kge_model is None:
            subject_output, relation_output, object_output = self._models
            sp_output = merge([subject_output, relation_output], mode='sum')


    def get_qa_model(self):
        if self._models is None:
            self._models = self.build()

        if self._qa_model is None:
            subject_output, relation_output, object_output = self._models
            sp_output = merge([subject_output, relation_output], mode='sum')

            similarity = self.get_similarity()
            qa_model = merge([sp_output, object_output], mode=similarity, output_shape=lambda x: x[:-1])

            self._qa_model = Model(input=[self.subject, self.relation, self.get_object()], output=[qa_model])

        return self._qa_model

    def get_qa_model_rt(self):
        if self._models is None:
            self._models = self.build()

        if self._qa_model_rt is None:
            subject_output, relation_output, object_output = self._models

            po_output = merge([object_output, relation_output], mode=lambda x: x[0] - x[1], output_shape=lambda x: x[0])

            similarity = self.get_similarity()
            qa_model_rt = merge([po_output, subject_output], mode=similarity, output_shape=lambda x: x[:-1])

            self._qa_model_rt = Model(input=[self.get_subject(), self.relation, self.object_good], output=[qa_model_rt])

        return self._qa_model_rt

    def compile(self, optimizer, **kwargs):
        qa_model = self.get_qa_model()

        good_output = qa_model([self.subject, self.relation, self.object_good])
        bad_output = qa_model([self.subject, self.relation, self.object_bad])

        loss = merge([good_output, bad_output],
                     mode=lambda x: K.maximum(1e-6, self.config['margin'] - x[0] + x[1]),
                     output_shape=lambda x: x[0])

        self.training_model = Model(input=[self.subject, self.relation, self.object_good, self.object_bad], output=loss)
        self.training_model.compile(loss=lambda y_true, y_pred: y_pred + y_true - y_true, optimizer=optimizer, **kwargs)

        self.prediction_model = Model(input=[self.subject, self.relation, self.object_good], output=good_output)
        self.prediction_model.compile(loss='binary_crossentropy', optimizer=optimizer, **kwargs)
        self.training_model.summary()

    def compile_rt(self, optimizer, **kwargs):
        qa_model_rt = self.get_qa_model_rt()

        good_output = qa_model_rt([self.subject, self.relation, self.object_good])
        bad_output = qa_model_rt([self.subject_bad, self.relation, self.object_good])

        loss = merge([good_output, bad_output],
                     mode=lambda x: K.maximum(1e-6, self.config['margin'] - x[0] + x[1]),
                     output_shape=lambda x: x[0])

        self.training_model_rt = Model(input=[self.subject, self.subject_bad, self.relation, self.object_good], output=loss)
        self.training_model_rt.compile(loss=lambda y_true, y_pred: y_pred + y_true - y_true, optimizer=optimizer, **kwargs)

        self.prediction_model_rt = Model(input=[self.subject, self.relation, self.object_good], output=good_output)
        self.prediction_model_rt.compile(loss='binary_crossentropy', optimizer=optimizer, **kwargs)

    def fit(self, x, **kwargs):
        assert self.training_model is not None, 'Must compile the model before fitting data'
        y = np.zeros(shape=x[0].shape[:1])
        return self.training_model.fit(x, y, **kwargs)

    def fit_rt(self, x, **kwargs):
        assert self.training_model_rt is not None, 'Must compile the model before fitting data'
        y = np.zeros(shape=x[0].shape[:1])
        return self.training_model_rt.fit(x, y, **kwargs)

    def train_on_batch(self, x, **kwargs):
        assert self.training_model is not None, 'Must compile the model before fitting data'
        y = np.zeros(shape=x[0].shape[:1])
        return self.training_model.train_on_batch(x, y, **kwargs)

    def train_on_batch_rt(self, x, **kwargs):
        assert self.training_model_rt is not None, 'Must compile the model before fitting data'
        y = np.zeros(shape=x[0].shape[:1])
        return self.training_model_rt.train_on_batch(x, y, **kwargs)

    def predict(self, x, **kwargs):
        return self.prediction_model.predict(x, **kwargs)

    def predict_rt(self, x, **kwargs):
        return self.prediction_model_rt.predict(x, **kwargs)

    def save_weights(self, file_name, **kwargs):
        assert self.prediction_model is not None, 'Must compile the model before saving weights'
        self.prediction_model.save_weights(file_name, **kwargs)

    def save_weights_rt(self, file_name, **kwargs):
        assert self.prediction_model_rt is not None, 'Must compile the model before saving weights'
        self.prediction_model_rt.save_weights(file_name, **kwargs)

    def load_weights(self, file_name, **kwargs):
        assert self.prediction_model is not None, 'Must compile the model loading weights'
        self.prediction_model.load_weights(file_name, **kwargs)

    def load_weights_rt(self, file_name, **kwargs):
        assert self.prediction_model_rt is not None, 'Must compile the model loading weights'
        self.prediction_model_rt.load_weights(file_name, **kwargs)











class HolE(Layer):
    def __init__(self, ndim=50, marge=1., lremb=0.1, lrparam=1., **kwargs):
        super().__init__(**kwargs)
        self.ndim = ndim
        self.marge = marge
        self.lremb = lremb
        self.lrparam = lrparam




import itertools
import logging
import numpy as np
import os
import time
import theano as th
import theano.tensor as T
from .gradient_descent import gd
from ..data_structures import triple_tensor as tt
from ..experiments.metrics import auprc
from .optimization import sgd_on_triples
from ..experiments.helper import tolist
_log = logging.getLogger(__name__)
DTYPE = th.config.floatX  # @UndefinedVariable
def init_uniform(rng, n, d, dtype=np.float32):
    wbound = np.sqrt(6. / d)
    W_values = rng.uniform(low=-wbound, high=wbound, size=(d, n))
    W_values = W_values / np.sqrt(np.sum(W_values ** 2, axis=0))
    W_values = np.asarray(W_values, dtype=dtype)
    return W_values.T
class TranslationalEmbeddingsModel(object):
    """Translational Embeddings Model.
    Implementation of TransE:
    Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, Oksana
    Yakhnenko. Translating Embeddings for Modeling Multi-relational Data.
    NIPS 2013
    Parameters
    ----------
    consider_tc : bool
        Whether or not to consider information about type constraints in the
        data.
        Defaults to True.
    simfn : string.
        'L1' or 'L2' similarity function.
        Defaults to 'L1'.
    ndim : int
        Dimension of the latent embeddings (rank).
        Defaults to 50.
    marge : float
        Margin in the margin based ranking function (gamma in the paper).
        Defaults to 1.
    lremb : float
        Learning rate for latent embeddings.
        Defaults to 0.1.
    lrparam : float
        Learning rate for other parameters.
        Defaults to 1.0.
    mbatchsize : int
        Size of the minibatch.
        Defaults to 128.
    totepoches : int
        Maximum epoches (how often the model is trained on the complete
        dataset).
        Defaults to 500.
    neval : int
        Validate performance every nth minibatch.
        Defaults to 1.
    lcwa : bool
        If true and consider_tc is True, approximate the type constraints from
        the data with the local closed-world assumption.
        Defaults to `False`.
    seed : int
        Seed used for random number generation.
        Defaults to 123.
    savepath : string
        Location where to save the best model parameters.
        Defaults to ./transE.
    """
    def __init__(self, consider_tc=True, simfn='L1', ndim=50, marge=1.,
                 lremb=0.1, lrparam=1., mbatchsize=128, maxepoch=500,
                 neval=100, lcwa=False, seed=123, conv=1e-4,
                 savepath='./transE', dtype=DTYPE,
                 mid=np.random.randint(1000000)):
        model_id = (time.strftime('%d_%m_%y___%H_%M_%S') +
                    '%d-%d_' % (mid, np.random.randint(100000)))
        self.simfn = simfn
        self.ndim = ndim
        self.marge = marge
        self.lremb = lremb
        self.lrparam = lrparam
        self.mbatchsize = mbatchsize
        self.maxepoch = maxepoch
        self.neval = neval
        self.seed = seed
        self.corrupted = 1
        self.corrupted_axes = [0, 1]
        self.rng = np.random.RandomState(seed)
        self.dtype = dtype
        self.consider_tc = consider_tc
        self.lcwa = lcwa
        self.conv = conv
        self.params = [ndim, marge, lremb, lrparam, simfn, seed, consider_tc,
                       lcwa]
        self.parallization_precautions = False
        self.savefile = os.path.join(savepath,
                                     model_id+type(self).__name__+".pkl")
        # create path where the model is saved
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
    def __graph_pred(self, X):
        # Translational Embeddings Function d(h+l,t)
        e = self.E[X[:, :2].T.reshape((-1,))]
        h = e[:e.shape[0]//2]
        l = self.R[X[:, 2]]
        t = e[e.shape[0]//2:]
        return (-T.sum(T.abs_((h+l)-t), axis=1)
                if self.simfn == 'L1'
                else - T.sqrt(T.sum(T.sqr((h+l)-t), axis=1)))
    def __graph_train(self, X, Xc):
        # Translational Embeddings max-margin loss function
        E = self.E[T.concatenate([X[:, :2], Xc[:, :2]],
                                 axis=1).T.reshape((-1,))]
        R = self.R[T.concatenate([X[:, 2], Xc[:, 2]])]
        e = E[:E.shape[0]//2]
        h = e[:e.shape[0]//2]
        l = R[:R.shape[0]//2]
        t = e[e.shape[0]//2:]
        outputX = (-T.sum(T.abs_((h+l)-t), axis=1)
                   if self.simfn == 'L1'
                   else - T.sqrt(T.sum(T.sqr((h+l)-t), axis=1)))
        ec = E[E.shape[0]//2:]
        hc = ec[:ec.shape[0]//2]
        lc = R[R.shape[0]//2:]
        tc = ec[ec.shape[0]//2:]
        outputXc = (-T.sum(T.abs_((hc+lc)-tc), axis=1)
                    if self.simfn == 'L1'
                    else - T.sqrt(T.sum(T.sqr((hc+lc)-tc), axis=1)))
        loss = outputXc - outputX + self.marge
        return T.sum(loss * (loss > 0))
    def loss_func(self, indices, Y):
        # Metric used for early stopping
        return 1-auprc(Y, self.func(indices))
    def fit(self, tensor):
        if not self.consider_tc:
            # remove type-constraint information
            tensor.type_constraints = [[None, None]
                                       for i in xrange(tensor.shape[2])]
        elif self.lcwa:
            tensor.approximate_type_constraints()
        self.type_constraints = tensor.type_constraints
        self.Nent = tensor.shape[0]
        self.Nrel = tensor.shape[2]
        self.samplefunc = tt.compute_corrupted_bordes
        X = T.imatrix("X")  # matrices with triple indices
        Xc = T.imatrix("Xc")  # corrupted entities
        self.E = th.shared(
            value=init_uniform(self.rng, tensor.shape[0], self.ndim,
                               dtype=self.dtype), name="Ents_emb")
        self.R = th.shared(
            value=init_uniform(self.rng, tensor.shape[0], self.ndim,
                               dtype=self.dtype), name="Rels_emb")
        self.parameters = [self.E, self.R]
        # Output function TransE: d(h+l,t)
        self.func = th.function([X], self.__graph_pred(X))
        # Define the cost function
        loss_pos = self.__graph_train(X, Xc)
        # Normalization function for embeddings of entities:
        batch_idcs = T.ivector('batch_idcs')
        update = OrderedDict({self.E: T.set_subtensor(
            self.E[batch_idcs], self.E[batch_idcs] /
            T.sqrt(T.sum(self.E[batch_idcs] ** 2, axis=1, keepdims=True)))})
        self.normalize = th.function([batch_idcs], [], updates=update)
        # Update function
        self.update_func = gd([X, Xc], loss_pos, self.parameters,
                              lr=[self.lremb,
                                  self.lrparam/float(self.mbatchsize)])
        # Train the model with stg
        fitted_parameters, self.used_epochs, self.epoch_times = (
            sgd_on_triples(self.rng, tensor, self, neval=self.neval,
                           mbsize=self.mbatchsize, unlabeled=True,
                           copy_X_train=not self.parallization_precautions))
        for i, parameter in enumerate(fitted_parameters):
            self.parameters[i].set_value(parameter.get_value())
    @property
    def sparsity(self):
        raise NotImplementedError
    def clear(self):
        """Deletes the memory expensive parameters."""
        del self.E
        del self.R
        del self.parameters
        os.remove(self.savefile)
    def predict(self, indices):
        # This should be just d(h+l,t)
        return self.func(indices)
    @staticmethod
    def model_creator(settings):
        # For loading multiple model parameters from a configuration file
        confs = None
        if settings['try_all_reg_combinations']:
            confs = list(itertools.product(tolist(settings['rank']),
                                           tolist(settings['gamma']),
                                           tolist(settings['lrate_emb']),
                                           tolist(settings['lrate_par'])))
        else:
            confs = [[r, m, lr1, lr2]
                     for r, m, lr1, lr2 in
                     zip(tolist(settings['rank']),
                         tolist(settings['gamma']),
                         tolist(settings['lrate_emb']),
                         tolist(settings['lrate_par']))]
        confs = list(itertools.product(tolist(settings['seed']), confs))
        models = []
        for i, conf in enumerate(confs):
            s, conf = conf
            r, m, lr1, lr2 = conf
            models.append(TranslationalEmbeddingsModel(
                consider_tc=settings['consider_tc'],
                simfn=str.upper(settings['simfn']),
                ndim=r,
                marge=m,
                lremb=lr1,
                lrparam=lr2,
                mbatchsize=settings['mbatchsize'],
                maxepoch=settings['maxepoch'],
                neval=settings['neval'],
                lcwa=settings['lcwa'],
                seed=s,
                savepath=settings['savepath'],
                mid=i))
        return models