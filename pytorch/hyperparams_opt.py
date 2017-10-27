from bayes_opt import BayesianOptimization

from kga.models import *
from kga.metrics import *
from kga.util import *
import numpy as np
import torch.optim
import argparse
import os
from time import time
from sklearn.utils import shuffle as skshuffle
from collections import defaultdict


parser = argparse.ArgumentParser(
    description='Train baselines methods: RESCAL, DistMult, ER-MLP, TransE'
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
parser.add_argument('--ntn_slice', type=int, default=4, metavar='',
                    help='number of slices used in NTN (default: 4)')
parser.add_argument('--mbsize', type=int, default=100, metavar='',
                    help='size of minibatch (default: 100)')
parser.add_argument('--negative_samples', type=int, default=10, metavar='',
                    help='number of negative samples per positive sample  (default: 10)')
parser.add_argument('--nepoch', type=int, default=5, metavar='',
                    help='number of training epoch (default: 5)')
parser.add_argument('--loss', default='logloss', metavar='',
                    help='loss function to be used, {"logloss", "rankloss"} (default: "logloss")')
parser.add_argument('--average_loss', default=False, action='store_true',
                    help='whether to average or sum the loss over minibatch')
parser.add_argument('--lr', type=float, default=0.1, metavar='',
                    help='learning rate (default: 0.1)')
parser.add_argument('--lr_decay_every', type=int, default=10, metavar='',
                    help='decaying learning rate every n epoch (default: 10)')
parser.add_argument('--weight_decay', type=float, default=1e-4, metavar='',
                    help='L2 weight decay (default: 1e-4)')
parser.add_argument('--embeddings_lambda', type=float, default=1e-2, metavar='',
                    help='prior strength for embeddings. Constraints embeddings norms to at most one  (default: 1e-2)')
parser.add_argument('--normalize_embed', default=False, type=bool, metavar='',
                    help='whether to normalize embeddings to unit euclidean ball (default: False)')
parser.add_argument('--use_gpu', default=False, type=bool, metavar='',
                    help='whether to run in the GPU or CPU (default: False <i.e. CPU>)')
parser.add_argument('--randseed', default=9999, type=int, metavar='',
                    help='resume the training from latest checkpoint (default: False)')
parser.add_argument('--n_iter', default=50, type=int, metavar='',
                    help='number of iteration of the hyperparams opt. (default: 50)')
parser.add_argument('--init_points', default=5, type=int, metavar='',
                    help='number of initial points before hyperparams opt. kicks in (default: 5)')
parser.add_argument('--start_over', default=False, action='store_true',
                    help='whether to start over the optimization or start from previous ones. (default: False, i.e. resume)')

args = parser.parse_args()


# Set random seed
np.random.seed(args.randseed)
torch.manual_seed(args.randseed)

if args.use_gpu:
    torch.cuda.manual_seed(args.randseed)


# Load dictionary lookups
idx2ent = np.load('data/NTN/{}/bin/idx2ent.npy'.format(args.dataset))
idx2rel = np.load('data/NTN/{}/bin/idx2rel.npy'.format(args.dataset))

n_e = len(idx2ent)
n_r = len(idx2rel)

# Load dataset
X_train = np.load('data/NTN/{}/bin/train.npy'.format(args.dataset))
X_val = np.load('data/NTN/{}/bin/val.npy'.format(args.dataset))
y_val = np.load('data/NTN/{}/bin/y_val.npy'.format(args.dataset))

X_val_pos = X_val[y_val.ravel() == 1, :]  # Take only positive samples

M_train = X_train.shape[0]
M_val = X_val.shape[0]


def evaluate_model(lr, decay, lam):
    lr, decay, lam = float(lr), float(decay), float(lam)

    C = args.negative_samples

    # Initialize model
    models = {
        'rescal': RESCAL(n_e=n_e, n_r=n_r, k=args.k, lam=lam, gpu=args.use_gpu),
        'distmult': DistMult(n_e=n_e, n_r=n_r, k=args.k, lam=lam, gpu=args.use_gpu),
        'ermlp': ERMLP(n_e=n_e, n_r=n_r, k=args.k, h_dim=args.mlp_h, p=args.mlp_dropout_p, lam=lam, gpu=args.use_gpu),
        'transe': TransE(n_e=n_e, n_r=n_r, k=args.k, gamma=args.transe_gamma, d=args.transe_metric, gpu=args.use_gpu),
        'ntn': NTN(n_e=n_e, n_r=n_r, k=args.k, lam=lam, slice=args.ntn_slice, gpu=args.use_gpu)
    }

    model = models[args.model]

    # Training params
    solver = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    n_epoch = args.nepoch
    mb_size = args.mbsize  # 2x with negative sampling

    # Begin training
    for epoch in range(n_epoch):
        # Shuffle and chunk data into minibatches
        mb_iter = get_minibatches(X_train, mb_size, shuffle=True)

        # Anneal learning rate
        lr = lr * (0.5 ** (epoch // args.lr_decay_every))
        for param_group in solver.param_groups:
            param_group['lr'] = lr

        for X_mb in mb_iter:
            # Build batch with negative sampling
            m = X_mb.shape[0]

            if args.loss == 'rankloss':
                # C x M negative samples
                X_neg_mb = np.vstack([sample_negatives(X_mb, n_e) for _ in range(C)])
            else:
                X_neg_mb = sample_negatives(X_mb, n_e)

            X_train_mb = np.vstack([X_mb, X_neg_mb])
            y_true_mb = np.vstack([np.ones([m, 1]), np.zeros([m, 1])])

            if args.loss == 'logloss':
                X_train_mb, y_true_mb = skshuffle(X_train_mb, y_true_mb)

            # Training step
            y = model.forward(X_train_mb)

            if args.loss == 'rankloss':
                y_pos, y_neg = y[:m], y[m:]

                loss = model.ranking_loss(
                    y_pos, y_neg, margin=args.transe_gamma, C=C, average=False
                )
            elif args.loss == 'logloss':
                loss = model.log_loss(y, y_true_mb, average=args.average_loss)

            loss.backward()
            solver.step()
            solver.zero_grad()

            if args.normalize_embed:
                model.normalize_embeddings()

    try:
        # Return the evaluation on validation set
        if args.loss == 'logloss':
            # Validation accuracy
            y_pred_val = model.forward(X_val)
            y_prob_val = F.sigmoid(y_pred_val)

            if args.use_gpu:
                val_auc = auc(y_prob_val.cpu().data.numpy(), y_val)
            else:
                val_auc = auc(y_prob_val.data.numpy(), y_val)

            return val_auc
        else:
            # For ranking loss, return hits@10
            mrr, hits10 = eval_embeddings(model, X_val_pos, n_e, k=10)

            return hits10
    except:
        # Error occurs, e.g. weights becomes NaN
        return 0


# ------------------------------
# Hyperparameters Optimization
# ------------------------------

opt_dir = 'hyperparams/{}'.format(args.dataset, args.model)

if not os.path.exists(opt_dir):
    os.makedirs(opt_dir)

num_iter = args.n_iter
init_points = args.init_points

opt = {
    'lr': (1e-6, 1),
    'decay': (1e-8, 1e-3),
    'lam': (1e-6, 1),
}

bo = BayesianOptimization(evaluate_model, opt)

# Load previous results
fname = '{}/{}.npy'.format(opt_dir, args.model)
init_dict = defaultdict(list)

# If previous results already exists, load it
if os.path.exists(fname):
    init_dict = np.load(fname).item()

if not args.start_over:
    bo.initialize(init_dict)

bo.maximize(init_points=init_points, n_iter=num_iter, acq='ucb')

# Append new infos to prior
for params, target in zip(bo.res['all']['params'], bo.res['all']['values']):
    init_dict['lr'].append(params['lr'])
    init_dict['decay'].append(params['decay'])
    init_dict['lam'].append(params['lam'])
    init_dict['target'].append(target)

np.save('{}/{}'.format(opt_dir, args.model), init_dict)
