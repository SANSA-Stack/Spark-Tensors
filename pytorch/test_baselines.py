from kga.models import *
from kga.metrics import *
from kga.util import *
import numpy as np
import torch.optim
import argparse


parser = argparse.ArgumentParser(
    description='Test baselines methods: RESCAL, DistMult, ER-MLP, TransE'
)

parser.add_argument('--model', default='rescal', metavar='',
                    help='model to run: {rescal, distmult, ermlp, transe} (default: rescal)')
parser.add_argument('--dataset', default='wordnet', metavar='',
                    help='dataset to be used: {wordnet, fb15k} (default: wordnet)')
parser.add_argument('--k', type=int, default=50, metavar='',
                    help='embedding dim (default: 50)')
parser.add_argument('--h', type=int, default=100, metavar='',
                    help='size of ER-MLP hidden layer (default: 100)')
parser.add_argument('--gamma', type=float, default=1, metavar='',
                    help='TransE loss margin (default: 1)')

args = parser.parse_args()

infos = {
    'wordnet': {
        'n_e': 40943,
        'n_r': 18
    },
    'fb15k': {
        'n_e': 14951,
        'n_r': 1345
    },
}

X_test, _, _ = load_data_bin('data/{}/bin/test.npy'.format(args.dataset))

n_e = infos[args.dataset]['n_e']
n_r = infos[args.dataset]['n_r']

models = {
    'rescal': RESCAL(n_e=n_e, n_r=n_r, k=args.k),
    'distmult': DistMult(n_e=n_e, n_r=n_r, k=args.k),
    'ermlp': ERMLP(n_e=n_e, n_r=n_r, k=args.k, h_dim=args.h),
    'transe': TransE(n_e=n_e, n_r=n_r, k=args.k, gamma=args.gamma)
}

model = models[args.model]
model.load_state_dict(torch.load('models/{}/{}.bin'.format(args.dataset, args.model)))

# Negative sampling
M_test = X_test.shape[1]

X_neg_test = sample_negatives(X_test, n_e=n_e)
X_all_test = np.hstack([X_test, X_neg_test])
y_true = np.vstack([np.ones([M_test, 1]), np.zeros([M_test, 1])])

# Test and show metrics
y_pred = model.predict(X_all_test)

print()
print('Test result for: {}'.format(args.model))
print('-----------------------------')

if args.model != 'transe':
    print('Accuracy: {}'.format(accuracy(y_pred, y_true)))
    print('AUC: {}'.format(auc(y_pred, y_true)))
else:
    print('MRR: {}, Hits@2: {}'.format(
        *eval_embeddings(model, X_test, n_e, k=args.k, mode='asc'))
    )
