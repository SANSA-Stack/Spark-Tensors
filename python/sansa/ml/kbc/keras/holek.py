import pickle
from random import shuffle
from keras.models import Model
from keras.layers import Input, Activation, Dense, Reshape
from keras.models import Sequential
import keras
from keras.layers import Layer
import math
import numpy as np
import keras.backend as K
from keras.optimizers import Adagrad
from theano import tensor as T
from sansa.ml.kbc.keras import sample

__author__ = 'nilesh'

class KerasHole(object):
    def __init__(self, numEntities, numRelations, ndim, rparam):
        self.numEntities = numEntities
        self.numRelations = numRelations
        self.ndim = ndim
        self.rparam = rparam

    def buildModel(self):
        inputs = Input(shape=(2,3))
        score = HolographicLayer2(self.numEntities, self.numRelations, self.ndim, self.rparam)(inputs)
        # score = Reshape((1,))(score)
        # score = Activation("sigmoid")(score)
        model = Model(input=inputs, output=score)
        adagrad = Adagrad(lr=0.001, epsilon=1e-06)

        def max_margin(y_true, y_pred):
            return T.sum(T.maximum(0., 1. + y_pred[1] + y_pred[0]))

        def loss(y_true, y_pred):
            # print(y_pred)
            return K.sum(K.log(1. + K.exp(-y_true * y_pred)))

        model.compile(optimizer='rmsprop', loss='binary_crossentropy')
        # Or try setting model's output=prediction and loss='binary_crossentropy' - essentially same thing as above
        return model

    def fit2(self, xs, ys):
        sampler = sample.RandomModeSampler(1, [0, 1], xs, (self.numEntities, self.numEntities, self.numRelations))
        xys = list(zip(xs, ys))
        xyns = sampler.sample(xys)
        shuffle(xys)
        shuffle(xyns)
        xs, ys = [np.array(i) for i in list(zip(*xys))]
        xns, yns = [np.array(i) for i in list(zip(*xyns))]
        # print(xs[:100], ys[:100])
        xpairs = [np.array(i) for i in list(zip(xs, xns))]
        ypairs = [np.array(i) for i in list(zip(ys, yns))]

        print (xpairs[0].shape)

        model = self.buildModel()
        # x = K.placeholder((3,))
        # func = K.function([x], model(x))
        # for x in xs:
        #     print(func([x]))
        best_weights_filepath = './best_weights.hdf5'
        # earlyStopping= keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
        saveBestModel = keras.callbacks.ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

        # train model
        history = model.fit(xpairs, ypairs, batch_size=len(xs)/1000, validation_split=0.1, nb_epoch=100,
                            callbacks=[saveBestModel])

        #reload best weights
        model.load_weights(best_weights_filepath)

        self.model = self
        self.E, self.R = model.layers[1].get_weights()

    def fit(self, xs, ys):
        sampler = sample.RandomModeSampler(1, [0, 1], xs, (self.numEntities, self.numEntities, self.numRelations))
        xys = list(zip(xs, ys))
        print(len(xys))
        xys += sampler.sample(xys)
        print(len(xys))
        shuffle(xys)
        xs, ys = [np.array(i) for i in list(zip(*xys))]
        # print(xs[:100], ys[:100])

        model = self.buildModel()
        # x = K.placeholder((3,))
        # func = K.function([x], model(x))
        # for x in xs:
        #     print(func([x]))
        best_weights_filepath = './best_weights.hdf5'
        # earlyStopping= keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
        saveBestModel = keras.callbacks.ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

        # train model
        history = model.fit(xs, ys, batch_size=len(xs)/1000, validation_split=0.05, nb_epoch=100,
                            callbacks=[saveBestModel])

        #reload best weights
        model.load_weights(best_weights_filepath)

        self.model = self
        self.E, self.R = model.layers[1].get_weights()


class HolographicLayer(Layer):
    def __init__(self, E, R, d, rparam, input_shape=(3,), **kwargs):
        from keras.initializations import glorot_normal
        self.init = [glorot_normal(shape=(E,d), name="E"), glorot_normal(shape=(R,d,d), name="R")]
        self.rparam = rparam
        kwargs["input_shape"] = input_shape
        super(HolographicLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        self.E, self.R = self.init
        self.trainable_weights = [self.E, self.R]
        from keras.regularizers import l2
        # regularizer = l2(self.rparam)
        # regularizer.set_param(self.E)
        # self.regularizers.append(regularizer)
        #
        # regularizer = l2(self.rparam)
        # regularizer.set_param(self.R)
        # self.regularizers.append(regularizer)

    def call(self, x, mask=None):
        batch_placeholder = K.cast(x, 'int32')[0]
        s, o, p = [batch_placeholder[i] for i in range(3)]

        s2v = K.gather(self.E, s)
        o2v = K.gather(self.E, o)
        r2v = K.gather(self.R, p)

        def ccorr(a, b):
            # Return tensor product - basically bilinear/RESCAL models
            return T.outer(a,b).flatten()

            # Or cross-correlation op?
            # return T.nnet.conv2d(a.dimshuffle('x', 'x', 0, 'x'), b.dimshuffle('x', 'x', 0, 'x'), None,
            #                None,
            #                filter_flip=True, border_mode='half').flatten()[:-1]
            # return self.ccorr1d_sc(a, b, border_mode='half')
        # eta = K.dot(r2v, ccorr(s2v, o2v))
        eta = K.dot(K.dot(s2v, r2v), o2v)

        # func = K.function([s2v,o2v,r2v], K.gradients(K.sigmoid(eta), [s2v,o2v,r2v]))
        # print(func([np.random.random(150),np.random.random(150),np.random.random(150)]))

        return eta

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], 1)

class HolographicLayer2(Layer):
    def __init__(self, E, R, d, rparam, input_shape=(2,3), **kwargs):
        from keras.initializations import glorot_normal
        self.init = [glorot_normal(shape=(E,d), name="E"), glorot_normal(shape=(R,d*d), name="R")]
        self.rparam = rparam
        kwargs["input_shape"] = input_shape
        super(HolographicLayer2, self).__init__(**kwargs)


    def build(self, input_shape):
        self.E, self.R = self.init
        self.trainable_weights = [self.E, self.R]
        from keras.regularizers import l2
        regularizer = l2(self.rparam)
        regularizer.set_param(self.E)
        self.regularizers.append(regularizer)

        regularizer = l2(self.rparam)
        regularizer.set_param(self.R)
        self.regularizers.append(regularizer)

    def call(self, x, mask=None):
        pos = K.cast(x, 'int32')[0][0]
        neg = K.cast(x, 'int32')[0][1]

        def eta(s, o, p):
            s2v = K.gather(self.E, s)
            o2v = K.gather(self.E, o)
            r2v = K.gather(self.R, p)

            def ccorr(a, b):
                # Return tensor product - basically bilinear/RESCAL models
                return T.outer(a,b).flatten()

                # Or cross-correlation op?
                # return T.nnet.conv2d(a.dimshuffle('x', 'x', 0, 'x'), b.dimshuffle('x', 'x', 0, 'x'), None,
                #                None,
                #                filter_flip=True, border_mode='half').flatten()[:-1]
                # return self.ccorr1d_sc(a, b, border_mode='half')
            eta = K.dot(r2v, ccorr(s2v, o2v))

            return eta


        pos_eta = eta(*[pos[i] for i in range(3)])
        neg_eta = eta(*[neg[i] for i in range(3)])
        return K.variable(np.array([pos_eta, neg_eta]))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], 2)

if __name__ == "__main__":
    wnbin = "/Users/nilesh/python/holographic-embeddings/data/wn18.bin"
    with open(wnbin, 'rb') as fin:
            data = pickle.load(fin)

    N = len(data['entities'])
    M = len(data['relations'])

    xs = data['train_subs']
    ys = np.ones(len(xs))

    trainer = KerasHole(N, M, 10, 0.01)
    trainer.fit2(xs, ys)