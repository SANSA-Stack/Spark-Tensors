import numpy as np
import scipy.spatial.distance
from sklearn.metrics import roc_auc_score
from collections import defaultdict


def accuracy(y_pred, y_true):
    """
    Compute accuracy score.

    Params:
    -------
    y_pred: np.array
        Predicted (Bernoulli) probabilities.

    y_true: np.array, binary
        True (Bernoulli) labels.
    """
    y = (y_pred >= 0.5)
    return np.mean(y == y_true)


def auc(y_pred, y_true):
    """
    Compute area under ROC curve score.

    Params:
    -------
    y_pred: np.array
        Predicted (Bernoulli) probabilities.

    y_true: np.array, binary
        True (Bernoulli) labels.
    """
    return roc_auc_score(y_true, y_pred)


def eval_embeddings(model, X_test, n_e, k, entity='head', mode='desc'):
    """
    Compute Mean Reciprocal Rank and Hits@k score of embedding model.
    The procedure follows Bordes, et. al., 2011.

    Params:
    -------
    model: kga.Model
        Embedding model to be evaluated.

    X_test: 3 x M matrix, where M is data size
        Contains M test triplets. First M/2 triplets are positive samples, while
        the rest M/2 triplets are negative samples.

    n_e: int
        Number of entities in dataset.

    k: int
        Max rank to be considered, i.e. to be used in Hits@k metric.

    entity: string {'head', 'tail'}, default 'head'
        Entity to be corrupted during evaluation.

    mode: string {'asc', 'desc'}, default 'desc'
        Whether to sort the score ascending (e.g. energy, distance) or
        descending (e.g. probability, score).

    Returns:
    --------
    mrr: float
        Mean Reciprocal Rank.

    hitsk: float
        Hits@k.
    """
    # Validate arguments
    if entity not in ('head', 'tail') or mode not in ('asc', 'desc'):
        raise ValueError('entity should be either "head" or "tail", '
                         'mode should be either "asc" or "desc"')

    M = X_test.shape[1]
    N = int(M/2)

    X_test_ori = np.copy(X_test[:, :N]).T
    scores = np.zeros([N, n_e])
    idx = 0 if entity == 'head' else 2  # Index of changed entity

    # Gather scores for all entities
    for e in range(n_e):
        X_test[idx, :] = e
        y = model.predict(X_test)
        scores[:, e] = y.ravel()

    # Sort scores in ascending order
    sorted_scores = np.argsort(scores)

    if mode == 'desc':
        # Reverse ordering if descending
        sorted_scores = sorted_scores[:, ::-1]

    # Compute ranks of original entities
    ranks = np.argmax(sorted_scores == X_test_ori[:, idx][:, None], axis=1)
    ranks += 1  # Convert to 1-indexed list

    # Compute mean reciprocal rank
    mrr = np.mean(1/ranks)

    # Compute hits@k
    hitsk = np.mean(ranks <= k)

    return mrr, hitsk


def entity_nn(model, n=10, k=5, idx2ent=None):
    """
    Compute nearest neighbours of all entities embeddings of a model.

    Params:
    -------
    model: instance of kga.Model

    n: int, default: 10
        Number of (random) entities to be queried.

    k: int, default: 5
        Number of nearest neighbours.

    idx2ent: dict, default: None
        Lookup dictionary to translate entity indices. If this is None, then
        output the indices matrix instead.
    """
    emb = model.emb_E.weight.data.numpy()
    mat = scipy.spatial.distance.cdist(emb, emb, metric='euclidean')
    idxs = np.random.randint(emb.shape[0], size=n)
    nn = np.argsort(mat, axis=1)[idxs, :k]

    if idx2ent is None:
        return nn

    nn_dict = defaultdict(dict)

    for i in nn:
        for j in i:
            k = idx2ent[i[0]]
            l = idx2ent[j]
            nn_dict[k][l] = mat[i[0], j]

    return dict(nn_dict)


def relation_nn(model, n=10, k=5, idx2rel=None):
    """
    Compute nearest neighbours of all relations embeddings of a model.

    Params:
    -------
    model: instance of kga.Model

    n: int, default: 10
        Number of (random) relations to be queried.

    k: int, default: 5
        Number of nearest neighbours.

    idx2rel: dict, default: None
        Lookup dictionary to translate relation indices. If this is None, then
        output the indices matrix instead.
    """
    emb = model.emb_L.weight.data.numpy()
    mat = scipy.spatial.distance.cdist(emb, emb, metric='euclidean')
    idxs = np.random.randint(emb.shape[0], size=n)
    nn = np.argsort(mat, axis=1)[idxs, :k]

    if idx2rel is None:
        return nn

    nn_dict = defaultdict(dict)

    for i in nn:
        for j in i:
            k = idx2rel[i[0]]
            l = idx2rel[j]
            nn_dict[k][l] = mat[i[0], j]

    return dict(nn_dict)
