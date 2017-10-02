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
parser.add_argument('--h', type=int, default=100, metavar='',
                    help='size of ER-MLP hidden layer (default: 100)')
parser.add_argument('--dropout_p', type=float, default=0.5, metavar='',
                    help='Probability of dropping out neuron in dropout (default: 0.5)')
parser.add_argument('--gamma', type=float, default=1, metavar='',
                    help='TransE loss margin (default: 1)')

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
    'ermlp': ERMLP(n_e=n_e, n_r=n_r, k=args.k, h_dim=args.h, p=args.dropout_p),
    'transe': TransE(n_e=n_e, n_r=n_r, k=args.k, gamma=args.gamma)
}

model = models[args.model]
model.load_state_dict(torch.load('models/{}/{}.bin'.format(args.dataset, args.model)))

# Test and show metrics
y_pred = model.predict(X_test)

print()
print('Test result for: {}'.format(args.model))
print('-----------------------------')

if args.model != 'transe':
    print('Accuracy: {:.4f}'.format(accuracy(y_pred, y_test)))
    print('AUC: {:.4f}'.format(auc(y_pred, y_test)))
    print(classification_report(y_test, (y_pred > 0.5)))
else:
    print('MRR: {}, Hits@2: {}'.format(
        *eval_embeddings(model, X_test, n_e, k=args.k, mode='asc'))
    )

nn_n = 10
nn_k = 5

idx2ent = np.load('data/NTN/{}/bin/idx2ent.npy'.format(args.dataset))
idx2rel = np.load('data/NTN/{}/bin/idx2rel.npy'.format(args.dataset))

print()
print('Entities nearest neighbours:')
print('----------------------------')

e_nn = entity_nn(model, n=nn_n, k=nn_k, idx2ent=idx2ent)
pprint(e_nn)

print()
print('Relations nearest neighbours:')
print('----------------------------')

r_nn = relation_nn(model, n=n_r, k=nn_k, idx2rel=idx2rel)
pprint(r_nn)
