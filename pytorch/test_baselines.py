from kga.models import *
from kga.metrics import *
from kga.util import *
import numpy as np
import torch.optim
import argparse
from sklearn.metrics import classification_report
from pprint import pprint


parser = argparse.ArgumentParser(
    description='Test baselines methods: RESCAL, DistMult, ER-MLP, TransE'
)

parser.add_argument('--model', default='rescal', metavar='',
                    help='model to run: {rescal, distmult, ermlp, transe} (default: rescal)')
parser.add_argument('--dataset', default='wordnet', metavar='',
                    help='dataset to be used: {wordnet, fb15k} (default: wordnet)')
parser.add_argument('--k', type=int, default=50, metavar='',
                    help='embedding dim (default: 50)')
parser.add_argument('--transe_gamma', type=float, default=1, metavar='',
                    help='TransE loss margin (default: 1)')
parser.add_argument('--transe_metric', default='l2', metavar='',
                    help='whether to use `l1` or `l2` metric for TransE (default: l2)')
parser.add_argument('--mlp_h', type=int, default=100, metavar='',
                    help='size of ER-MLP hidden layer (default: 100)')
parser.add_argument('--mlp_dropout_p', type=float, default=0.5, metavar='',
                    help='Probability of dropping out neuron in dropout (default: 0.5)')
parser.add_argument('--hit_k', type=int, default=10, metavar='',
                    help='hit@k metrics (default: 10)')
parser.add_argument('--nn_n', type=int, default=5, metavar='',
                    help='number of entities/relations for nearest neighbours (default: 5)')
parser.add_argument('--nn_k', type=int, default=5, metavar='',
                    help='k in k-nearest-neighbours (default: 5)')

args = parser.parse_args()

infos = {
    'wordnet': {
        'n_e': 38194,
        'n_r': 11
    },
    'fb15k': {
        'n_e': 75043,
        'n_r': 13
    },
}

X_test, _, _ = load_data_bin('data/NTN/{}/bin/test.npy'.format(args.dataset))
y_test = np.load('data/NTN/{}/bin/y_test.npy'.format(args.dataset))

n_e = infos[args.dataset]['n_e']
n_r = infos[args.dataset]['n_r']

models = {
    'rescal': RESCAL(n_e=n_e, n_r=n_r, k=args.k),
    'distmult': DistMult(n_e=n_e, n_r=n_r, k=args.k),
    'ermlp': ERMLP(n_e=n_e, n_r=n_r, k=args.k, h_dim=args.mlp_h, p=args.mlp_dropout_p),
    'transe': TransE(n_e=n_e, n_r=n_r, k=args.k, gamma=args.transe_gamma, d=args.transe_metric)
}

model = models[args.model]
model.load_state_dict(torch.load('models/{}/{}.bin'.format(args.dataset, args.model)))


# Evaluation metrics, e.g. acc, auc, mrr, hits@k
# ----------------------------------------------

print()
print('Test result for: {}'.format(args.model))
print('-----------------------------')

if args.model != 'transe':
    y_pred = model.predict(X_test)

    print('Accuracy: {:.4f}'.format(accuracy(y_pred, y_test)))
    print('AUC: {:.4f}'.format(auc(y_pred, y_test)))
    print(classification_report(y_test, (y_pred > 0.5)))
else:
    # Only take positive samples
    X_pos = X_test[:, y_test.ravel() == 1]
    mrr, hitsk = eval_embeddings(model, X_pos, n_e, k=args.hit_k)

    print('MRR: {:.4f}, Hits@{}: {:.4f}'.format(mrr, args.hit_k, hitsk))


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
