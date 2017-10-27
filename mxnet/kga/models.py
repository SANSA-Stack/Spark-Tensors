import numpy as np

import mxnet as mx
import mxnet.gluon as gluon
from mxnet.gluon import nn
from mxnet import autograd
from mxnet import nd

import kga.op as op
import kga.util as util
from kga.util import inherit_docstrings


class TransE(gluon.Block):
    """
    TransE embedding model
    ----------------------
    Bordes, Antoine, et al.
    "Translating embeddings for modeling multi-relational data." NIPS. 2013.
    """

    def __init__(self, n_e, n_r, k, d='l2', gpu=False):
        """
        TransE embedding model
        ----------------------

        Params:
        -------
            n_e: int
                Number of entities in dataset.

            n_r: int
                Number of relationships in dataset.

            k: int
                Embedding size.

            d: {'l1', 'l2'}
                Distance measure to be used in the loss.

            gpu: bool, default: False
                Whether to use GPU or not.
        """
        super(TransE, self).__init__()

        # Hyperparams
        self.n_e = n_e  # Num of entities
        self.n_r = n_r  # Num of rels
        self.k = k
        self.d = d

        self.ctx = mx.gpu() if gpu else mx.cpu()

        # Embeddings, initialized as in the paper
        self.emb_E = nn.Embedding(self.n_e, self.k)
        self.emb_R = nn.Embedding(self.n_r, self.k)

        init = mx.initializer.Uniform(6/np.sqrt(k))
        self.emb_E.initialize(init, self.ctx)
        self.emb_R.initialize(init, self.ctx)

        self.embeddings = [self.emb_E, self.emb_R]
        self.normalize_embeddings()

        # Remove relation embeddings from list so that it won't be normalized
        # during training.
        self.embeddings = [self.emb_E]

    def normalize_embeddings(self):
        for e in self.embeddings:
            # Take norm of each entity/relation in the embedding matrix
            weight = e.weight.data()
            norm = nd.sqrt(nd.sum(weight**2, axis=1))

            # Normalize
            e.weight.set_data(weight / nd.expand_dims(norm, 1))

    def forward(self, X):
        """
        Given a (mini)batch of triplets X of size M, compute the energies.

        Params:
        -------
        X: int matrix of M x 3, where M is the (mini)batch size
            First column contains index of head entities.
            Second column contains index of relationships.
            Third column contains index of tail entities.

        Returns:
        --------
        f: float matrix of M x 1
            Contains energies of each triplets.
        """
        # Decompose X into head, relationship, tail
        hs, ls, ts = X[:, 0], X[:, 1], X[:, 2]

        hs = nd.array(hs, self.ctx)
        ls = nd.array(ls, self.ctx)
        ts = nd.array(ts, self.ctx)

        e_hs = self.emb_E(hs)
        e_ts = self.emb_E(ts)
        e_ls = self.emb_R(ls)

        f = self.energy(e_hs, e_ls, e_ts).reshape([-1, 1])

        return f

    def energy(self, h, l, t):
        """
        Compute TransE energy

        Params:
        -------
        h: Mxk tensor
            Contains head embeddings.

        l: Mxk tensor
            Contains relation embeddings.

        t: Mxk tensor
            Contains tail embeddings.

        Returns:
        --------
        E: Mx1 tensor
            Energy of each triplets, computed by d(h + l, t) for some func d.
        """
        if self.d == 'l1':
            out = nd.sum(nd.abs(h + l - t), axis=1)
        else:
            out = nd.sqrt(nd.sum((h + l - t)**2, axis=1))

        return out

    def ranking_loss(self, y_pos, y_neg, margin=1, C=1):
        """
        Compute loss max margin ranking loss.

        Params:
        -------
        y_pos: vector of size Mx1
            Contains scores for positive samples.

        y_neg: np.array of size Mx1 (binary)
            Contains the true labels.

        margin: float, default: 1
            Margin used for the loss.

        C: int, default: 1
            Number of negative samples per positive sample.

        Returns:
        --------
        loss: scalar
        """
        y_pos = nd.repeat(y_pos.reshape([-1]), C)  # repeat to match y_neg
        y_neg = y_neg.reshape([-1])

        loss = nd.sum(nd.relu(margin + y_pos - y_neg))

        return loss

    def predict(self, X):
        """
        Predict the score of test batch.

        Params:
        -------
        X: int matrix of M x 3, where M is the (mini)batch size
            First row contains index of head entities.
            Second row contains index of relationships.
            Third row contains index of tail entities.

        Returns:
        --------
        y_pred: np.array of Mx1
        """
        y_pred = self.forward(X).reshape([-1, 1])
        return y_pred.asnumpy()
