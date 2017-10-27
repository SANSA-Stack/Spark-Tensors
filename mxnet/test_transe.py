from kga.models import *
from kga.metrics import *
from kga.util import *
import numpy as np

import mxnet as mx
import mxnet.gluon as gluon
from mxnet.gluon import nn
from mxnet import autograd
from mxnet import nd

import argparse
from sklearn.metrics import classification_report
from pprint import pprint


parser = argparse.ArgumentParser(
    description='Test TransE'
)

parser.add_argument('--dataset', default='wordnet', metavar='',
                    help='dataset to be used: {wordnet, fb15k} (default: wordnet)')
parser.add_argument('--k', type=int, default=50, metavar='',
                    help='embedding dim (default: 50)')
parser.add_argument('--transe_gamma', type=float, default=1, metavar='',
                    help='TransE loss margin (default: 1)')
parser.add_argument('--transe_metric', default='l2', metavar='',
                    help='whether to use `l1` or `l2` metric for TransE (default: l2)')
parser.add_argument('--hit_k', type=int, default=10, metavar='',
                    help='hit@k metrics (default: 10)')
parser.add_argument('--nn_n', type=int, default=5, metavar='',
                    help='number of entities/relations for nearest neighbours (default: 5)')
parser.add_argument('--nn_k', type=int, default=5, metavar='',
                    help='k in k-nearest-neighbours (default: 5)')
parser.add_argument('--use_gpu', default=False, type=bool, metavar='',
                    help='whether to run in the GPU or CPU (default: False <i.e. CPU>)')
parser.add_argument('--randseed', default=9999, type=int, metavar='',
                    help='resume the training from latest checkpoint (default: False')

args = parser.parse_args()


# Set random seed
np.random.seed(args.randseed)
mx.random.seed(args.randseed)

ctx = mx.gpu() if args.use_gpu else mx.cpu()

# Load dictionary lookups
idx2ent = np.load('data/NTN/{}/bin/idx2ent.npy'.format(args.dataset))
idx2rel = np.load('data/NTN/{}/bin/idx2rel.npy'.format(args.dataset))

n_e = len(idx2ent)
n_r = len(idx2rel)

# Load val data
X_val = np.load('data/NTN/{}/bin/val.npy'.format(args.dataset))
y_val = np.load('data/NTN/{}/bin/y_val.npy'.format(args.dataset))

# Load test data
X_test = np.load('data/NTN/{}/bin/test.npy'.format(args.dataset))
y_test = np.load('data/NTN/{}/bin/y_test.npy'.format(args.dataset))

checkpoint_dir = '{}/{}'.format(args.checkpoint_dir.rstrip('/'), args.dataset)
checkpoint_path = '{}/{}.bin'.format(checkpoint_dir, 'transe')

# Initialize model
model = TransE(n_e, n_r, args.k, args.transe_metric, args.use_gpu)
model.collect_params().load(checkpoint_path)


# Evaluation metrics, e.g. acc, auc, mrr, hits@k
# ----------------------------------------------

print()
print('Test result for: {}'.format(args.model))
print('-----------------------------')

# Only take positive samples
X_pos = X_test[y_test.ravel() == 1, :]
mrr, hitsk = eval_embeddings(model, X_pos, n_e, k=args.hit_k)

print('MRR: {:.4f}'.format(mrr))
print('Hits@{}: {:.4f}'.format(args.hit_k, hitsk))

# Find the best thresholds for TransE using the validation set
y_pred = model.predict(X_test)

acc = 0

for r in range(n_r):
    # Get data where the relation == r
    idxs = (X_val[:, 1] == r)
    X_val_r = X_val[idxs, :]
    y_val_r = y_val[idxs, :]

    # Find the best thresholds for this relation
    thresh = find_clf_threshold(model, X_val_r, y_val_r)

    # Use the threshold on test set
    idxs = (X_test[:, 1] == r)
    X_test_r = X_test[idxs, :]
    y_test_r = y_test[idxs, :]
    y_pred_r = y_pred[idxs, :]

    y = (y_pred_r >= thresh)
    res = np.mean(y == y_test_r)

    acc += res

print('Accuracy: {:.4f}'.format(acc/n_r))


# Nearest-neighbours
# ------------------

idx2ent = np.load('data/NTN/{}/bin/idx2ent.npy'.format(args.dataset))
idx2rel = np.load('data/NTN/{}/bin/idx2rel.npy'.format(args.dataset))

print()
print('Entities nearest neighbours:')
print('----------------------------')

e_nn = entity_nn(model, n=args.nn_n, k=args.nn_k, idx2ent=idx2ent)
pprint(e_nn)

print()
print('Relations nearest neighbours:')
print('----------------------------')

r_nn = relation_nn(model, n=args.nn_n, k=args.nn_k, idx2rel=idx2rel)
pprint(r_nn)
