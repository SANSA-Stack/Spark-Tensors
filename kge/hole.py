

from __future__ import print_function

from collections import OrderedDict
import keras
import theano as th
import theano.tensor as T

from keras import backend as K
from keras.optimizers import Adagrad, SGD
import keras
from keras.engine.topology import Layer
from keras.models import Sequential, Model
from keras.layers import merge, Input, Embedding, Dropout, Convolution1D, Lambda, Activation, LSTM, Dense, TimeDistributed, \
    ActivityRegularization, Reshape, Flatten
from keras.constraints import unitnorm

import os
import sys
import random
import numpy as np
from time import strftime, gmtime
import six.moves.cPickle as pickle
from keras.optimizers import RMSprop, Adam, SGD, Adadelta, Adagrad
from scipy.stats import rankdata

__author__ = 'nilesh'

class KgeModel:
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
        self._kge_model = None

        self.training_model = None
        self.prediction_model = None

    def get_object(self):
        if self._object is None:
            self._object = Input(shape=(self.config['object_len'],), dtype='int32', name='object')
        return self._object

    def get_subject(self):
        if self._subject is None:
            self._subject = Input(shape=(self.config['subject_len'],), dtype='int32', name='subject')
        return self._subject

    # @abstractmethod
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

            # relation_output2 = Reshape((100,100))(relation_output)
            sp_output = merge([subject_output, relation_output], mode='sum')
            # so_output = merge([subject_output, object_output], mode=lambda x: np.outer(x[0], x[1]).reshape(100000,))
            spo_output = merge([sp_output, Reshape((0,100))(object_output)], mode=lambda a, b: K.batch_dot(a, b, axes=len(a._keras_shape) - 1),
                               output_shape=lambda x: x[0])

            self._kge_model = Model(input=[self.subject, self.relation, self.get_object()], output=[spo_output])
        return self._kge_model


    def compile(self, optimizer, **kwargs):
        kge_model = self.get_kge_model()

        good_output = kge_model([self.subject, self.relation, self.object_good])
        bad_output = kge_model([self.subject, self.relation, self.object_bad])

        loss = merge([good_output, bad_output],
                     mode=lambda x: K.maximum(1e-6, self.config['margin'] - x[0] + x[1]),
                     output_shape=lambda x: x[0])

        self.training_model = Model(input=[self.subject, self.relation, self.object_good, self.object_bad], output=loss)
        self.training_model.compile(loss=lambda y_true, y_pred: y_pred + y_true - y_true, optimizer=optimizer, **kwargs)

        self.prediction_model = Model(input=[self.subject, self.relation, self.object_good], output=good_output)
        self.prediction_model.compile(loss='binary_crossentropy', optimizer=optimizer, **kwargs)
        self.training_model.summary()

    def fit(self, x, **kwargs):
        assert self.training_model is not None, 'Must compile the model before fitting data'
        y = np.zeros(shape=x[0].shape[:1])
        return self.training_model.fit(x, y, **kwargs)


    def train_on_batch(self, x, **kwargs):
        assert self.training_model is not None, 'Must compile the model before fitting data'
        y = np.zeros(shape=x[0].shape[:1])
        return self.training_model.train_on_batch(x, y, **kwargs)

    def predict(self, x, **kwargs):
        return self.prediction_model.predict(x, **kwargs)

    def save_weights(self, file_name, **kwargs):
        assert self.prediction_model is not None, 'Must compile the model before saving weights'
        self.prediction_model.save_weights(file_name, **kwargs)

    def load_weights(self, file_name, **kwargs):
        assert self.prediction_model is not None, 'Must compile the model loading weights'
        self.prediction_model.load_weights(file_name, **kwargs)




class RescalModel(KgeModel):
    def build(self):
        subject = self.subject
        relation = self.relation
        object_ = self.get_object()
        embedding_size = self.model_params.get('n_embed_dims', 100)

        # add embedding layers
        embedding_rel = Embedding(input_dim=self.config['n_rel'],
                                  output_dim=self.model_params.get('n_embed_dims', 100),
                                  init='he_uniform',
                                  mask_zero=False)
        embedding_ent = Embedding(input_dim=self.config['n_ent'],
                                  output_dim=self.model_params.get('n_embed_dims', 100),
                                  init='he_uniform',
                                  W_constraint=unitnorm(axis=1),
                                  mask_zero=False)
        subject_embedding = embedding_ent(subject)
        relation_embedding = embedding_rel(relation)
        object_embedding = embedding_ent(object_)

        subject_output = Reshape((embedding_size,))(subject_embedding)
        relation_output = Reshape((embedding_size,))(relation_embedding)
        object_output = Reshape((embedding_size,))(object_embedding)

        return subject_output, relation_output, object_output









random.seed(42)
os.environ['FREEBASE_15K'] = 'data/freebase15k'


class Evaluator:
    def __init__(self, conf=None):
        try:
            data_path = os.environ['FREEBASE_15K']
        except KeyError:
            print("FREEBASE_15K is not set.")
            sys.exit(1)
        self.path = data_path
        self.conf = dict() if conf is None else conf
        self.params = conf.get('training_params', dict())
        self.entity = self.load('freebase_15k-id2entity.pkl')
        self._vocab = None
        self._reverse_vocab = None
        self._eval_sets = None

    ##### Resources #####

    def load(self, name):
        return pickle.load(open(os.path.join(self.path, name), 'rb'))

    def vocab(self):
        if self._vocab is None:
            self._vocab = self.load('vocabulary')
        return self._vocab

    def reverse_vocab(self):
        if self._reverse_vocab is None:
            vocab = self.vocab()
            self._reverse_vocab = dict((v.lower(), k) for k, v in vocab.items())
        return self._reverse_vocab

    ##### Loading / saving #####

    def save_epoch(self, model, epoch):
        if not os.path.exists('models/freebase_models/embedding/'):
            os.makedirs('models/freebase_models/embedding/')
        model.save_weights('models/freebase_models/embedding/weights_epoch_%d.h5' % epoch, overwrite=True)

    def load_epoch(self, model, epoch):
        assert os.path.exists('models/freebase_models/embedding/weights_epoch_%d.h5' % epoch),\
            'Weights at epoch %d not found' % epoch
        model.load_weights('models/freebase_models/embedding/weights_epoch_%d.h5' % epoch)

    ##### Converting / reverting #####

    def convert(self, words):
        rvocab = self.reverse_vocab()
        if type(words) == str:
            words = words.strip().lower().split(' ')
        return [rvocab.get(w, 0) for w in words]

    def revert(self, indices):
        vocab = self.vocab()
        return [vocab.get(i, 'X') for i in indices]

    ##### Padding #####

    def padq(self, data):
        return self.pad(data, self.conf.get('question_len', None))

    def pada(self, data):
        return self.pad(data, self.conf.get('answer_len', None))

    def pad(self, data, len=None):
        from keras.preprocessing.sequence import pad_sequences
        return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)

    ##### Training #####

    def print_time(self):
        print(strftime('%Y-%m-%d %H:%M:%S :: ', gmtime()), end='')

    def train(self, model):
        eval_every = self.params.get('eval_every', None)
        save_every = self.params.get('save_every', None)
        batch_size = self.params.get('batch_size', 128)
        nb_epoch = self.params.get('nb_epoch', 10)
        split = self.params.get('validation_split', 0)

        training_set = self.load('freebase_15k-train.pkl')
        valid_set = self.load('freebase_15k-valid.pkl')

        subjects = list()
        relations = list()
        good_objects = list()

        for line in training_set:
            triplet = line.split('\t')
            subjects += [[int(triplet[0])]]
            relations += [[int(triplet[1])]]
            good_objects += [[int(triplet[2])]]

        subjects = np.asarray(subjects)
        relations = np.asarray(relations)
        good_objects = np.asarray(good_objects)

        # subjects_valid = list()
        # relations_valid = list()
        # good_objects_valid = list()
        #
        # for line in valid_set:
        #     triplet = line.split('\t')
        #     subjects_valid += [[int(triplet[0])]]
        #     relations_valid += [[int(triplet[1])]]
        #     good_objects_valid += [[int(triplet[2])]]

        # subjects_valid = np.asarray(subjects_valid)
        # relations_valid = np.asarray(relations_valid)
        # good_objects_valid = np.asarray(good_objects_valid)

        val_loss = {'loss': 1., 'epoch': 0}

        for i in range(1, nb_epoch+1):
            # bad_answers = np.roll(good_answers, random.randint(10, len(questions) - 10))
            # bad_answers = good_answers.copy()
            # random.shuffle(bad_answers)
            bad_objects = np.asarray([[int(random.choice(self.entity.keys()))] for _ in xrange(len(good_objects))])

            # shuffle questionsj
            # zipped = zip(questions, good_answers)
            # random.shuffle(zipped)
            # questions[:], good_answers[:] = zip(*zipped)

            print('Epoch %d :: ' % i, end='')
            self.print_time()
            model.fit([subjects, relations, good_objects, bad_objects], nb_epoch=1, batch_size=batch_size)

            # if hist.history['val_loss'][0] < val_loss['loss']:
            #     val_loss = {'loss': hist.history['val_loss'][0], 'epoch': i}
            # print('Best: Loss = {}, Epoch = {}'.format(val_loss['loss'], val_loss['epoch']))

            if eval_every is not None and i % eval_every == 0:
                self.get_mrr(model)

            if save_every is not None and i % save_every == 0:
                self.save_epoch(model, i)

    ##### Evaluation #####

    def prog_bar(self, so_far, total, n_bars=20):
        n_complete = int(so_far * n_bars / total)
        if n_complete >= n_bars - 1:
            print('\r[' + '=' * n_bars + ']', end='')
        else:
            s = '\r[' + '=' * (n_complete - 1) + '>' + '.' * (n_bars - n_complete) + ']'
            print(s, end='')

    def eval_sets(self):
        if self._eval_sets is None:
            self._eval_sets = dict([(s, self.load(s)) for s in ['freebase_15k-test.pkl']])
        return self._eval_sets

    def get_mrr(self, model, evaluate_all=False):
        top1s = list()
        mrrs = list()
        for name, data in self.eval_sets().items():
            if evaluate_all:
                self.print_time()
                print('----- %s -----' % name)

            random.shuffle(data)

            if not evaluate_all and 'n_eval' in self.params:
                data = data[:self.params['n_eval']]

            # c_1 for hit@1, c_3 for hit@3, c_10 for hit@10
            c_1, c_3, c_10 = 0, 0, 0
            mean_ranks = list()

            for i, d in enumerate(data):
                triplet = d.split('\t')
                if evaluate_all:
                    self.prog_bar(i, len(data))

                candidate_objects = self.entity.keys()
                candidate_objects.remove(int(triplet[2]))

                subject = np.asarray([[int(triplet[0])]] * (len(candidate_objects)+1))
                relation = np.asarray([[int(triplet[1])]] * (len(candidate_objects)+1))
                objects = np.asarray([[int(triplet[2])]] + [[entity_id] for entity_id in candidate_objects])
                sims = model.predict([subject, relation, objects], batch_size=len(self.entity)).flatten()
                r = rankdata(sims, method='max')

                target_rank = r[0]
                num_candidate = len(sims)
                real_rank = num_candidate - target_rank + 1

                # print(' '.join(self.revert(d['question'])))
                # print(' '.join(self.revert(self.answers[indices[max_r]])))
                # print(' '.join(self.revert(self.answers[indices[max_n]])))

                c_1 += 1 if target_rank == num_candidate else 0
                c_3 += 1 if target_rank + 3 > num_candidate else 0
                c_10 += 1 if target_rank + 10 > num_candidate else 0
                mean_ranks.append(real_rank)
                # c_2 += 1 / float(r[max_r] - r[max_n] + 1)

            hit_at_1 = c_1 / float(len(data))
            hit_at_3 = c_3 / float(len(data))
            hit_at_10 = c_10 / float(len(data))
            avg_rank = np.mean(mean_ranks)

            del data

            if evaluate_all:
                print('Hit@1 Precision: %f' % hit_at_1)
                print('Hit@3 Precision: %f' % hit_at_3)
                print('Hit@10 Precision: %f' % hit_at_10)
                print('Mean Rank: %f' % avg_rank)

            # top1s.append(top1)
            # mrrs.append(mrr)

        # rerun the evaluation if above some threshold
        if not evaluate_all:
            print('Top-1 Precision: {}'.format(top1s))
            print('MRR: {}'.format(mrrs))
            evaluate_all_threshold = self.params.get('evaluate_all_threshold', dict())
            evaluate_mode = evaluate_all_threshold.get('mode', 'all')
            mrr_theshold = evaluate_all_threshold.get('mrr', 1)
            top1_threshold = evaluate_all_threshold.get('top1', 1)

            if evaluate_mode == 'any':
                evaluate_all = evaluate_all or any([x >= top1_threshold for x in top1s])
                evaluate_all = evaluate_all or any([x >= mrr_theshold for x in mrrs])
            else:
                evaluate_all = evaluate_all or all([x >= top1_threshold for x in top1s])
                evaluate_all = evaluate_all or all([x >= mrr_theshold for x in mrrs])

            if evaluate_all:
                return self.get_mrr(model, evaluate_all=True)

if __name__ == '__main__':
    conf = {
        'subject_len': 1,
        'relation_len': 1,
        'object_len': 1,
        'n_rel': 1345,  # len(vocabulary)
        'n_ent': 14951,
        'margin': 0.2,

        'training_params': {
            'save_every': 100,
            # 'eval_every': 1,
            'batch_size': 128,
            'nb_epoch': 1000,
            'validation_split': 0,
            'optimizer': Adam(),
            # 'optimizer': Adam(clip_norm=0.1),
            # 'n_eval': 100,

            'evaluate_all_threshold': {
                'mode': 'all',
                'top1': 0.4,
            },
        },

        'model_params': {
            'n_embed_dims': 100,
            'n_hidden': 200,

            # convolution
            'nb_filters': 1000, # * 4
            'conv_activation': 'relu',

            # recurrent
            'n_lstm_dims': 141, # * 2

            # 'initial_embed_weights': np.load('models/wordnet_word2vec_1000_dim.h5'),
        },

        'similarity_params': {
            'mode': 'cosine',
            'gamma': 1,
            'c': 1,
            'd': 2,
        }
    }

    evaluator = Evaluator(conf)

    ##### Embedding model ######
    model = RescalModel(conf)
    optimizer = conf.get('training_params', dict()).get('optimizer', 'adam')

    # TransE model
    # model = TranEModel(conf)
    # optimizer = conf.get('training_params', dict()).get('optimizer', 'adam')

    model.compile(optimizer=optimizer)

    # save embedding layer
    # evaluator.load_epoch(model, 33)
    # embedding_layer = model.prediction_model.layers[2].layers[2]
    # evaluator.load_epoch(model, 100)
    # evaluator.train(model)
    # weights = embedding_layer.get_weights()[0]
    # np.save(open('models/embedding_1000_dim.h5', 'wb'), weights)

    # train the model
    # evaluator.load_epoch(model, 54)
    evaluator.train(model)
    # embedding_matrix = model.prediction_model.layers[3].layers[3].get_weights()[0]
    # print(np.linalg.norm(embedding_matrix[1, :]))
    # print(np.linalg.norm(embedding_matrix[:, 1]))

    # evaluate mrr for a particular epoch
    # evaluator.load_epoch(model, 5)
    # evaluator.get_mrr(model, evaluate_all=True)










# class HolE(Layer):
#     def __init__(self, ndim=50, marge=1., lremb=0.1, lrparam=1., **kwargs):
#         super().__init__(**kwargs)
#         self.ndim = ndim
#         self.marge = marge
#         self.lremb = lremb
#         self.lrparam = lrparam




# import itertools
# import logging
# import numpy as np
# import os
# import time
# import theano as th
# import theano.tensor as T
# from .gradient_descent import gd
# from ..data_structures import triple_tensor as tt
# from ..experiments.metrics import auprc
# from .optimization import sgd_on_triples
# from ..experiments.helper import tolist
# _log = logging.getLogger(__name__)
# DTYPE = th.config.floatX  # @UndefinedVariable
# def init_uniform(rng, n, d, dtype=np.float32):
#     wbound = np.sqrt(6. / d)
#     W_values = rng.uniform(low=-wbound, high=wbound, size=(d, n))
#     W_values = W_values / np.sqrt(np.sum(W_values ** 2, axis=0))
#     W_values = np.asarray(W_values, dtype=dtype)
#     return W_values.T
# class TranslationalEmbeddingsModel(object):
#     """Translational Embeddings Model.
#     Implementation of TransE:
#     Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, Oksana
#     Yakhnenko. Translating Embeddings for Modeling Multi-relational Data.
#     NIPS 2013
#     Parameters
#     ----------
#     consider_tc : bool
#         Whether or not to consider information about type constraints in the
#         data.
#         Defaults to True.
#     simfn : string.
#         'L1' or 'L2' similarity function.
#         Defaults to 'L1'.
#     ndim : int
#         Dimension of the latent embeddings (rank).
#         Defaults to 50.
#     marge : float
#         Margin in the margin based ranking function (gamma in the paper).
#         Defaults to 1.
#     lremb : float
#         Learning rate for latent embeddings.
#         Defaults to 0.1.
#     lrparam : float
#         Learning rate for other parameters.
#         Defaults to 1.0.
#     mbatchsize : int
#         Size of the minibatch.
#         Defaults to 128.
#     totepoches : int
#         Maximum epoches (how often the model is trained on the complete
#         dataset).
#         Defaults to 500.
#     neval : int
#         Validate performance every nth minibatch.
#         Defaults to 1.
#     lcwa : bool
#         If true and consider_tc is True, approximate the type constraints from
#         the data with the local closed-world assumption.
#         Defaults to `False`.
#     seed : int
#         Seed used for random number generation.
#         Defaults to 123.
#     savepath : string
#         Location where to save the best model parameters.
#         Defaults to ./transE.
#     """
#     def __init__(self, consider_tc=True, simfn='L1', ndim=50, marge=1.,
#                  lremb=0.1, lrparam=1., mbatchsize=128, maxepoch=500,
#                  neval=100, lcwa=False, seed=123, conv=1e-4,
#                  savepath='./transE', dtype=DTYPE,
#                  mid=np.random.randint(1000000)):
#         model_id = (time.strftime('%d_%m_%y___%H_%M_%S') +
#                     '%d-%d_' % (mid, np.random.randint(100000)))
#         self.simfn = simfn
#         self.ndim = ndim
#         self.marge = marge
#         self.lremb = lremb
#         self.lrparam = lrparam
#         self.mbatchsize = mbatchsize
#         self.maxepoch = maxepoch
#         self.neval = neval
#         self.seed = seed
#         self.corrupted = 1
#         self.corrupted_axes = [0, 1]
#         self.rng = np.random.RandomState(seed)
#         self.dtype = dtype
#         self.consider_tc = consider_tc
#         self.lcwa = lcwa
#         self.conv = conv
#         self.params = [ndim, marge, lremb, lrparam, simfn, seed, consider_tc,
#                        lcwa]
#         self.parallization_precautions = False
#         self.savefile = os.path.join(savepath,
#                                      model_id+type(self).__name__+".pkl")
#         # create path where the model is saved
#         if not os.path.isdir(savepath):
#             os.mkdir(savepath)
#     def __graph_pred(self, X):
#         # Translational Embeddings Function d(h+l,t)
#         e = self.E[X[:, :2].T.reshape((-1,))]
#         h = e[:e.shape[0]//2]
#         l = self.R[X[:, 2]]
#         t = e[e.shape[0]//2:]
#         return (-T.sum(T.abs_((h+l)-t), axis=1)
#                 if self.simfn == 'L1'
#                 else - T.sqrt(T.sum(T.sqr((h+l)-t), axis=1)))
#     def __graph_train(self, X, Xc):
#         # Translational Embeddings max-margin loss function
#         E = self.E[T.concatenate([X[:, :2], Xc[:, :2]],
#                                  axis=1).T.reshape((-1,))]
#         R = self.R[T.concatenate([X[:, 2], Xc[:, 2]])]
#         e = E[:E.shape[0]//2]
#         h = e[:e.shape[0]//2]
#         l = R[:R.shape[0]//2]
#         t = e[e.shape[0]//2:]
#         outputX = (-T.sum(T.abs_((h+l)-t), axis=1)
#                    if self.simfn == 'L1'
#                    else - T.sqrt(T.sum(T.sqr((h+l)-t), axis=1)))
#         ec = E[E.shape[0]//2:]
#         hc = ec[:ec.shape[0]//2]
#         lc = R[R.shape[0]//2:]
#         tc = ec[ec.shape[0]//2:]
#         outputXc = (-T.sum(T.abs_((hc+lc)-tc), axis=1)
#                     if self.simfn == 'L1'
#                     else - T.sqrt(T.sum(T.sqr((hc+lc)-tc), axis=1)))
#         loss = outputXc - outputX + self.marge
#         return T.sum(loss * (loss > 0))
#     def loss_func(self, indices, Y):
#         # Metric used for early stopping
#         return 1-auprc(Y, self.func(indices))
#     def fit(self, tensor):
#         if not self.consider_tc:
#             # remove type-constraint information
#             tensor.type_constraints = [[None, None]
#                                        for i in xrange(tensor.shape[2])]
#         elif self.lcwa:
#             tensor.approximate_type_constraints()
#         self.type_constraints = tensor.type_constraints
#         self.Nent = tensor.shape[0]
#         self.Nrel = tensor.shape[2]
#         self.samplefunc = tt.compute_corrupted_bordes
#         X = T.imatrix("X")  # matrices with triple indices
#         Xc = T.imatrix("Xc")  # corrupted entities
#         self.E = th.shared(
#             value=init_uniform(self.rng, tensor.shape[0], self.ndim,
#                                dtype=self.dtype), name="Ents_emb")
#         self.R = th.shared(
#             value=init_uniform(self.rng, tensor.shape[0], self.ndim,
#                                dtype=self.dtype), name="Rels_emb")
#         self.parameters = [self.E, self.R]
#         # Output function TransE: d(h+l,t)
#         self.func = th.function([X], self.__graph_pred(X))
#         # Define the cost function
#         loss_pos = self.__graph_train(X, Xc)
#         # Normalization function for embeddings of entities:
#         batch_idcs = T.ivector('batch_idcs')
#         update = OrderedDict({self.E: T.set_subtensor(
#             self.E[batch_idcs], self.E[batch_idcs] /
#             T.sqrt(T.sum(self.E[batch_idcs] ** 2, axis=1, keepdims=True)))})
#         self.normalize = th.function([batch_idcs], [], updates=update)
#         # Update function
#         self.update_func = gd([X, Xc], loss_pos, self.parameters,
#                               lr=[self.lremb,
#                                   self.lrparam/float(self.mbatchsize)])
#         # Train the model with stg
#         fitted_parameters, self.used_epochs, self.epoch_times = (
#             sgd_on_triples(self.rng, tensor, self, neval=self.neval,
#                            mbsize=self.mbatchsize, unlabeled=True,
#                            copy_X_train=not self.parallization_precautions))
#         for i, parameter in enumerate(fitted_parameters):
#             self.parameters[i].set_value(parameter.get_value())
#     @property
#     def sparsity(self):
#         raise NotImplementedError
#     def clear(self):
#         """Deletes the memory expensive parameters."""
#         del self.E
#         del self.R
#         del self.parameters
#         os.remove(self.savefile)
#     def predict(self, indices):
#         # This should be just d(h+l,t)
#         return self.func(indices)
#     @staticmethod
#     def model_creator(settings):
#         # For loading multiple model parameters from a configuration file
#         confs = None
#         if settings['try_all_reg_combinations']:
#             confs = list(itertools.product(tolist(settings['rank']),
#                                            tolist(settings['gamma']),
#                                            tolist(settings['lrate_emb']),
#                                            tolist(settings['lrate_par'])))
#         else:
#             confs = [[r, m, lr1, lr2]
#                      for r, m, lr1, lr2 in
#                      zip(tolist(settings['rank']),
#                          tolist(settings['gamma']),
#                          tolist(settings['lrate_emb']),
#                          tolist(settings['lrate_par']))]
#         confs = list(itertools.product(tolist(settings['seed']), confs))
#         models = []
#         for i, conf in enumerate(confs):
#             s, conf = conf
#             r, m, lr1, lr2 = conf
#             models.append(TranslationalEmbeddingsModel(
#                 consider_tc=settings['consider_tc'],
#                 simfn=str.upper(settings['simfn']),
#                 ndim=r,
#                 marge=m,
#                 lremb=lr1,
#                 lrparam=lr2,
#                 mbatchsize=settings['mbatchsize'],
#                 maxepoch=settings['maxepoch'],
#                 neval=settings['neval'],
#                 lcwa=settings['lcwa'],
#                 seed=s,
#                 savepath=settings['savepath'],
#                 mid=i))
#         return models