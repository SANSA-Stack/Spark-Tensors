import pickle
from random import shuffle
from keras.models import Model
from keras.layers import Input, Activation, Dense, Reshape
from keras.models import Sequential
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
        inputs = Input(shape=(3,))
        score = HolographicLayer(self.numEntities, self.numRelations, self.ndim, self.rparam)(inputs)
        # score = Reshape((1,))(score)
        prediction = Activation("sigmoid")(score)
        model = Model(input=inputs, output=score)
        adagrad = Adagrad(lr=0.01, epsilon=1e-06)

        def loss(y_true, y_pred):
            # print(y_pred)
            return -K.sum(K.log(K.sigmoid(-y_true * y_pred)))

        model.compile(optimizer=adagrad, loss=loss)
        # Or try setting model's output=prediction and loss='binary_crossentropy' - essentially same thing as above
        return model

    def fit(self, xs, ys):
        sampler = sample.RandomModeSampler(1, [0, 1], xs, (self.numEntities, self.numEntities, self.numRelations))
        xys = list(zip(xs, ys))
        xys += sampler.sample(xys)
        shuffle(xys)
        model = self.buildModel()
        model.fit(xs, ys, batch_size=len(xs)/100, nb_epoch=100)
        self.model = model


class HolographicLayer(Layer):
    def __init__(self, E, R, d, rparam, input_shape=(3,), **kwargs):
        bnd = math.sqrt(6) / math.sqrt(2*E)
        from numpy.random import uniform
        self.init = [K.variable(uniform(size=(E,d), low=-bnd, high=bnd), name="E"),
                     K.variable(uniform(size=(R,d*d), low=-bnd, high=bnd), name="R")]
        self.rparam = rparam
        kwargs["input_shape"] = input_shape
        super(HolographicLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        self.E, self.R = self.init
        self.trainable_weights = [self.E, self.R]
        # from keras.regularizers import l2
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
            return T.outer(a,b).flatten()
            # return self.ccorr1d_sc(a, b, border_mode='half')
        eta = K.dot(r2v, ccorr(s2v, o2v))

        # func = K.function([s2v,o2v,r2v], K.gradients(K.sigmoid(eta), [s2v,o2v,r2v]))
        # print(func([np.random.random(150),np.random.random(150),np.random.random(150)]))

        return eta

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], 1)


if __name__ == "__main__":
    wnbin = "/Users/nilesh/python/holographic-embeddings/data/wn18.bin"
    with open(wnbin, 'rb') as fin:
            data = pickle.load(fin)

    N = len(data['entities'])
    M = len(data['relations'])

    xs = data['train_subs']
    ys = np.ones(len(xs))

    trainer = KerasHole(N, M, 100, 0)
    trainer.fit(xs, ys)