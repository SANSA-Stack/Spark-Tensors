import numpy as np
import pandas as pd
from inspect import getmembers, isfunction
from sklearn.utils import shuffle as skshuffle
from time import time


def sample_negatives(X, n_e):
    """
    Perform negative sampling by corrupting head or tail of each triplets in
    dataset.

    Params:
    -------
    X: int matrix of M x 3, where M is the (mini)batch size
        First column contains index of head entities.
        Second column contains index of relationships.
        Third column contains index of tail entities.

    n_e: int
        Number of entities in dataset.

    Returns:
    --------
    X_corr: int matrix of M x 3, where M is the (mini)batch size
        Similar to input param X, but at each column, either first or third col
        is subtituted with random entity.
    """
    M = X.shape[0]

    corr = np.random.randint(n_e, size=M)
    e_idxs = np.random.choice([0, 2], size=M)

    X_corr = np.copy(X)
    X_corr[np.arange(M), e_idxs] = corr

    return X_corr


def sample_negatives2(X, n_e):
    """
    Perform negative sampling by corrupting head or tail of each triplets in
    dataset.
    In this function, the replacement entities will be guaranteed to be
    different to the original entities.

    Params:
    -------
    X: int matrix of M x 3, where M is the (mini)batch size
        First column contains index of head entities.
        Second column contains index of relationships.
        Third column contains index of tail entities.

    n_e: int
        Number of entities in dataset.

    Returns:
    --------
    X_corr: int matrix of M x 3, where M is the (mini)batch size
        Similar to input param X, but at each row, either first or third column
        is subtituted with random entity.
    """
    M = X.shape[0]
    X_corr = []

    for x in X:
        h, t = x[0], x[2]

        hc = np.random.randint(n_e)
        while hc == h: hc = np.random.randint(n_e)

        r = x[1]

        tc = np.random.randint(n_e)
        while tc == t: tc = np.random.randint(n_e)

        X_corr.append([hc, r, t])
        X_corr.append([h, r, tc])

    return np.array(X_corr, dtype=int)


def load_data(file_path):
    """
    Load raw dataset into tensor of indexes.

    Params:
    -------
    file_path: string
        Path to the dataset file. The dataset should be CSV with 3 columns
        separated by \t.

    Returns:
    --------
    X: np.array of M x 3
        Contains the triplets from dataset. The entities and relations are
        translated to its unique indices.

    y: [Only if the dataset contains this information] binary np.array of Mx1
        Class label of each M data.

    n_e: int
        Total number of unique entities in the dataset.

    n_r: int
        Total number of unique relations in the dataset.

    idx2ent: list
        Lookup table to recover entity name from its index.

    idx2rel: list
        Lookup table to recover relation name from its index.
    """
    df = pd.read_csv(file_path, sep='\t', header=None)

    # Get unique entities
    entities = pd.concat([df[0], df[2]]).unique()
    # Get unique relations
    relations = df[1].unique()

    M = df.shape[0]  # dataset size
    n_e = entities.shape[0]  # num of entities
    n_r = relations.shape[0]  # num of relations

    idx2ent = entities.tolist()
    idx2rel = relations.tolist()

    ent2idx = {e: idx for idx, e in enumerate(entities)}
    rel2idx = {r: idx for idx, r in enumerate(relations)}

    X = np.zeros([M, 3], dtype=int)

    for i, row in df.iterrows():
        X[i, 0] = ent2idx[row[0]]
        X[i, 1] = rel2idx[row[1]]
        X[i, 2] = ent2idx[row[2]]

    # Check if labels exists
    if df.shape[1] >= 4:
        y = df[3].values
        return X, y, n_e, n_r, idx2ent, idx2rel
    else:
        return X, n_e, n_r, idx2ent, idx2rel


def load_data_bin(file_path):
    """
    Load processed and pickled dataset into tensor of indexes.

    Params:
    -------
    file_path: string
        Path to the pickled dataset file. The dataset should be .npy file.

    Returns:
    --------
    X: np.array of M x 3
        Contains the triplets from dataset. The entities and relations are
        translated to its unique indices.

    n_e: int
        Total number of unique entities in the dataset.

    n_r: int
        Total number of unique relations in the dataset.
    """
    X = np.load(file_path)
    n_e = max(np.max(X[:, 0]), np.max(X[:, 2])) + 1
    n_r = np.max(X[:, 1]) + 1
    return X, int(n_e), int(n_r)


def get_minibatches(X, mb_size, shuffle=True):
    """
    Generate minibatches from given dataset for training.

    Params:
    -------
    X: np.array of M x 3
        Contains the triplets from dataset. The entities and relations are
        translated to its unique indices.

    mb_size: int
        Size of each minibatch.

    shuffle: bool, default True
        Whether to shuffle the dataset before dividing it into minibatches.

    Returns:
    --------
    mb_iter: generator
        Example usage:
        --------------
        mb_iter = get_minibatches(X_train, mb_size)
        for X_mb in mb_iter:
            // do something with X_mb, the minibatch
    """
    minibatches = []
    X_shuff = np.copy(X)

    if shuffle:
        X_shuff = skshuffle(X_shuff)

    for i in range(0, X_shuff.shape[0], mb_size):
        yield X_shuff[i:i + mb_size]


def get_random_minibatch(X, mb_size):
    idxs = np.random.choice(np.arange(X.shape[0]), size=mb_size, replace=False)
    return X[idxs, :]


def inherit_docstrings(cls):
    """
    Decorator to inherit docstring of class/method
    """
    for name, func in getmembers(cls, isfunction):
        if func.__doc__:
            continue

        parent = cls.__mro__[1]

        if hasattr(parent, name):
            func.__doc__ = getattr(parent, name).__doc__

    return cls
