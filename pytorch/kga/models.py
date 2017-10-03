import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import kga.op as op
import kga.util as util
from kga.util import inherit_docstrings


class Model(nn.Module):
    """
    Base class of all models
    """

    def __init__(self):
        super(Model, self).__init__()
        self.embeddings = []

    def forward(self, X):
        """
        Given a (mini)batch of triplets X of size M, predict the validity.

        Params:
        -------
        X: int matrix of 3 x M, where M is the (mini)batch size
            First row contains index of head entities.
            Second row contains index of relationships.
            Third row contains index of tail entities.

        Returns:
        --------
        y: Mx1 vectors
            Contains the probs result of each M data.
        """
        raise NotImplementedError

    def predict(self, X):
        """
        Predict the score of test batch.

        Params:
        -------
        X: int matrix of 3 x M, where M is the (mini)batch size
            First row contains index of head entities.
            Second row contains index of relationships.
            Third row contains index of tail entities.

        Returns:
        --------
        y_pred: np.array of Mx1
        """
        return self.forward(X).data.numpy()

    def loss(self, X):
        """
        Compute loss.

        Params:
        -------
        y_pred: vector of size Mx1
            Contains prediction probabilities.

        y_true: np.array of size Mx1 (binary)
            Contains the true labels.

        Returns:
        --------
        loss: float
        """
        raise NotImplementedError

    def normalize_embeddings(self):
        for e in self.embeddings:
            e.weight.data.renorm_(p=2, dim=0, maxnorm=1)


@inherit_docstrings
class ERLMLP(Model):
    """
    ERL-MLP: Entity-Relation-Literal MLP
    ------------------------------------
    """

    def __init__(self, n_e, n_r, n_a, k, l, h_dim):
        """
        ERL-MLP: Entity-Relation-Literal MLP
        ------------------------------------

        Params:
        -------
            n_e: int
                Number of entities in dataset.

            n_r: int
                Number of relationships in dataset.

            n_a: int
                Number of attributes/literals in dataset.

            k: int
                Embedding size for entity and relationship.

            l: int
                Size of projected attributes/literals.

            h_dim: int
                Size of hidden layer.
        """
        super(ERLMLP, self).__init__()

        # Hyperparams
        self.n_e = n_e
        self.n_r = n_r
        self.n_a = n_a
        self.k = k
        self.l = l
        self.h_dim = h_dim

        # Nets
        self.emb_E = nn.Embedding(self.n_e, self.k)
        self.emb_R = nn.Embedding(self.n_r, self.k)
        self.fc_literal = nn.Linear(self.n_a, self.l)
        self.fc1 = nn.Linear(3*k+l, h_dim)
        self.fc2 = nn.Linear(h_dim, 1)

        self.embeddings = [self.emb_E]

        # Initialize embeddings
        r = 6/np.sqrt(k)

        self.emb_E.weight.data.uniform_(-r, r)
        self.emb_R.weight.data.uniform_(-r, r)

        # Normalize rel embeddings
        self.emb_E.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.emb_R.weight.data.renorm_(p=2, dim=0, maxnorm=1)

    def forward(self, X, X_lit):
        """
        Given a (mini)batch of triplets X of size M, predict the validity.

        Params:
        -------
        X: int matrix of 3 x M, where M is the (mini)batch size
            First row contains index of head entities.
            Second row contains index of relationships.
            Third row contains index of tail entities.

        X_lit: float matrix of M x n_a
            Contains all literals/attributes information of all data in batch.
            i-th row correspond to the i-th data in X.

        Returns:
        --------
        y: Mx1 vectors
            Contains the probs result of each M data.
        """
        # Decompose X into head, relationship, tail
        hs, ls, ts = X

        hs = Variable(torch.from_numpy(hs))
        ls = Variable(torch.from_numpy(ls))
        ts = Variable(torch.from_numpy(ts))
        X_lit = Variable(torch.from_numpy(X_lit))

        # Project to embedding, each is M x k
        e_hs = self.emb_E(hs)
        e_ts = self.emb_E(ts)
        e_ls = self.emb_R(ls)

        # Project literals to lower dimension subspace
        e_as = self.fc_literal(X_lit)

        # Forward
        phi = torch.cat([e_hs, e_ts, e_ls, e_as], 1)  # M x 3k
        h = F.relu(self.fc1(phi))
        y_logit = self.fc2(h)
        y_prob = F.sigmoid(y_logit)

        return y_prob

    def predict(self, X, X_lit):
        """
        Predict the score of test batch.

        Params:
        -------
        X: int matrix of 3 x M, where M is the (mini)batch size
            First row contains index of head entities.
            Second row contains index of relationships.
            Third row contains index of tail entities.

        X_lit: float matrix of M x n_a
            Contains all literals/attributes information of all data in batch.
            i-th row correspond to the i-th data in X.

        Returns:
        --------
        y_pred: np.array of Mx1
        """
        return self.forward(X, X_lit).data.numpy()

    def loss(self, y_pred, y_true):
        y_true = Variable(torch.from_numpy(y_true.astype(np.float32)))
        return F.binary_cross_entropy(y_pred, y_true)


@inherit_docstrings
class RESCAL(Model):
    """
    RESCAL: bilinear model
    ----------------------
    Nickel, Maximilian, Volker Tresp, and Hans-Peter Kriegel.
    "A three-way model for collective learning on multi-relational data."
    ICML. 2011.
    """

    def __init__(self, n_e, n_r, k):
        """
        RESCAL: bilinear model
        ----------------------

        Params:
        -------
            n_e: int
                Number of entities in dataset.

            n_r: int
                Number of relationships in dataset.

            k: int
                Embedding size.
        """
        super(RESCAL, self).__init__()

        # Hyperparams
        self.n_e = n_e
        self.n_r = n_r
        self.k = k

        # Nets
        self.emb_E = nn.Embedding(self.n_e, self.k)
        self.emb_R = nn.Embedding(self.n_r, self.k**2)

        self.embeddings = [self.emb_E, self.emb_R]

        # Initialize embeddings
        r1 = 6/np.sqrt(k)
        r2 = 6/k

        self.emb_E.weight.data.uniform_(-r1, r1)
        self.emb_R.weight.data.uniform_(-r2, r2)

        # Normalize rel embeddings
        self.emb_E.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.emb_R.weight.data.renorm_(p=2, dim=0, maxnorm=1)

    def forward(self, X):
        # Decompose X into head, relationship, tail
        hs, ls, ts = X

        hs = Variable(torch.from_numpy(hs))
        ls = Variable(torch.from_numpy(ls))
        ts = Variable(torch.from_numpy(ts))

        # Project to embedding, each is M x k
        e_hs = self.emb_E(hs).view(-1, self.k, 1)
        e_ts = self.emb_E(ts).view(-1, self.k, 1)
        W = self.emb_R(ls).view(-1, self.k, self.k)  # M x k x k

        # Forward
        out = torch.bmm(torch.transpose(e_hs, 1, 2), W)  # h^T W
        out = torch.bmm(out, e_ts)  # (h^T W) h
        out = out.view(-1, 1)  # [-1, 1, 1] -> [-1, 1]

        y_prob = F.sigmoid(out)
        y_prob = y_prob.view(-1, 1)  # Reshape to Mx1

        return y_prob

    def loss(self, y_pred, y_true):
        y_true = Variable(torch.from_numpy(y_true.astype(np.float32)))
        return F.binary_cross_entropy(y_pred, y_true)


@inherit_docstrings
class DistMult(Model):
    """
    DistMult: diagonal bilinear model
    ---------------------------------
    Yang, Bishan, et al. "Learning multi-relational semantics using
    neural-embedding models." arXiv:1411.4072 (2014).
    """

    def __init__(self, n_e, n_r, k):
        """
        DistMult: diagonal bilinear model
        ---------------------------------

        Params:
        -------
            n_e: int
                Number of entities in dataset.

            n_r: int
                Number of relationships in dataset.

            k: int
                Embedding size.
        """
        super(DistMult, self).__init__()

        # Hyperparams
        self.n_e = n_e
        self.n_r = n_r
        self.k = k

        # Nets
        self.emb_E = nn.Embedding(self.n_e, self.k)
        self.emb_R = nn.Embedding(self.n_r, self.k)

        self.embeddings = [self.emb_E, self.emb_R]

        # Initialize embeddings
        r = 6/np.sqrt(k)

        self.emb_E.weight.data.uniform_(-r, r)
        self.emb_R.weight.data.uniform_(-r, r)

        # Normalize rel embeddings
        self.emb_E.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.emb_R.weight.data.renorm_(p=2, dim=0, maxnorm=1)

    def forward(self, X):
        # Decompose X into head, relationship, tail
        hs, ls, ts = X

        hs = Variable(torch.from_numpy(hs))
        ls = Variable(torch.from_numpy(ls))
        ts = Variable(torch.from_numpy(ts))

        # Project to embedding, each is M x k
        e_hs = self.emb_E(hs)
        e_ts = self.emb_E(ts)
        W = self.emb_R(ls)

        # Forward
        f = torch.sum(e_hs * W * e_ts, 1)
        y_prob = F.sigmoid(f)
        y_prob = y_prob.view(-1, 1)  # Reshape to Mx1

        return y_prob

    def loss(self, y_pred, y_true):
        y_true = Variable(torch.from_numpy(y_true.astype(np.float32)))
        return F.binary_cross_entropy(y_pred, y_true)


@inherit_docstrings
class ERMLP(Model):
    """
    ER-MLP: Entity-Relation MLP
    ---------------------------
    """

    def __init__(self, n_e, n_r, k, h_dim, p):
        """
        ER-MLP: Entity-Relation MLP
        ---------------------------

        Params:
        -------
            n_e: int
                Number of entities in dataset.

            n_r: int
                Number of relationships in dataset.

            k: int
                Embedding size.

            h_dim: int
                Size of hidden layer.

            p: float
                Dropout rate.
        """
        super(ERMLP, self).__init__()

        # Hyperparams
        self.n_e = n_e
        self.n_r = n_r
        self.k = k
        self.h_dim = h_dim
        self.p = p

        # Nets
        self.emb_E = nn.Embedding(self.n_e, self.k)
        self.emb_R = nn.Embedding(self.n_r, self.k)

        self.mlp = nn.Sequential(
            nn.BatchNorm1d(3*k),
            nn.Dropout(p=self.p),
            nn.Linear(3*k, h_dim),
            nn.ReLU(),
            nn.BatchNorm1d(h_dim),
            nn.Dropout(p=self.p),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )

        self.embeddings = [self.emb_E, self.emb_R]

        # Initialize embeddings
        r = 6/np.sqrt(k)

        self.emb_E.weight.data.uniform_(-r, r)
        self.emb_R.weight.data.uniform_(-r, r)

        # Normalize rel embeddings
        self.emb_R.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.emb_E.weight.data.renorm_(p=2, dim=0, maxnorm=1)

        # Xavier init
        for p in self.mlp.modules():
            if isinstance(p, nn.Linear):
                in_dim = p.weight.size(0)
                p.weight.data.normal_(0, 1/np.sqrt(in_dim/2))

    def forward(self, X):
        # Decompose X into head, relationship, tail
        hs, ls, ts = X

        hs = Variable(torch.from_numpy(hs))
        ls = Variable(torch.from_numpy(ls))
        ts = Variable(torch.from_numpy(ts))

        # Project to embedding, each is M x k
        e_hs = self.emb_E(hs)
        e_ts = self.emb_E(ts)
        e_ls = self.emb_R(ls)

        # Forward
        phi = torch.cat([e_hs, e_ts, e_ls], 1)  # M x 3k
        y_prob = self.mlp(phi)

        return y_prob

    def loss(self, y_pred, y_true):
        y_true = Variable(torch.from_numpy(y_true.astype(np.float32)))
        return F.binary_cross_entropy(y_pred, y_true)


@inherit_docstrings
class TransE(Model):
    """
    TransE embedding model
    ----------------------
    Bordes, Antoine, et al.
    "Translating embeddings for modeling multi-relational data." NIPS. 2013.
    """

    def __init__(self, n_e, n_r, k, gamma, d='l2'):
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

            gamma: float
                Margin size for TransE's hinge loss.

            d: {'l1', 'l2'}
                Distance measure to be used in the loss.
        """
        super(TransE, self).__init__()

        # Hyperparams
        self.n_e = n_e  # Num of entities
        self.n_r = n_r  # Num of rels
        self.k = k
        self.gamma = gamma
        self.d = d

        # Nets
        self.emb_E = nn.Embedding(self.n_e, self.k)
        self.emb_R = nn.Embedding(self.n_r, self.k)

        self.embeddings = [self.emb_E]

        # Initialize embeddings
        r = 6/np.sqrt(k)

        self.emb_E.weight.data.uniform_(-r, r)
        self.emb_R.weight.data.uniform_(-r, r)

        # Normalize rel embeddings
        self.emb_E.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.emb_R.weight.data.renorm_(p=2, dim=0, maxnorm=1)

    def forward(self, X):
        """
        Given a (mini)batch of triplets X of size M, create corrupted triplets,
        and compute all the embeddings.

        Params:
        -------
        X: int matrix of 3 x M, where M is the (mini)batch size
            First row contains index of head entities.
            Second row contains index of relationships.
            Third row contains index of tail entities.
            First M/2 columns are positive data, the rest are negative samples.

        Returns:
        --------
        y: tuple of 5 embeddings
            Contains the embeddings of (order matters!): head, rel, tail,
            corrupted head, corrupted tail.
        """
        M = X.shape[1]

        # Negative sampling
        X_neg = X[:, int(M/2):]

        # Decompose X into head, relationship, tail
        hs, ls, ts = X[:, :int(M/2)]
        hcs, tcs = X_neg[0], X_neg[2]

        hs = Variable(torch.from_numpy(hs))
        ls = Variable(torch.from_numpy(ls))
        ts = Variable(torch.from_numpy(ts))
        hcs = Variable(torch.from_numpy(hcs))
        tcs = Variable(torch.from_numpy(tcs))

        e_hs = self.emb_E(hs)
        e_ts = self.emb_E(ts)
        e_ls = self.emb_R(ls)
        e_hcs = self.emb_E(hcs)
        e_tcs = self.emb_E(tcs)

        return e_hs, e_ls, e_ts, e_hcs, e_tcs

    def predict(self, X):
        y = self.forward(X)
        return self.energy(y).data.numpy()

    def loss(self, y, y_true=None):
        return torch.sum(self.energy(y))

    def energy(self, y):
        """
        Compute TransE energy

        Params:
        -------
        y: tuple of 5 embeddings
            Contains the embeddings of (order matters!): head, rel, tail,
            corrupted head, corrupted tail.

        Returns:
        --------
        E: Mx1 tensor
            E = relu(gamma + d1 - d2), where d1 is Lp-norm computed using
            real triplets and d2 is computed using corrupted triplets.
        """
        h, l, t, hc, tc = y

        if self.d == 'l1':
            x_pos = torch.abs(h + l - t)
            x_neg = torch.abs(hc + l - tc)
        else:
            x_pos = (h + l - t)**2
            x_neg = (hc + l - tc)**2

        d_pos = torch.sum(x_pos, 1)
        d_neg = torch.sum(x_neg, 1)

        if self.d == 'l2':
            d_pos = torch.sqrt(d_pos)
            d_neg = torch.sqrt(d_neg)

        return F.relu(self.gamma + d_pos - d_neg).view(-1, 1)
