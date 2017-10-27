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
import os
from time import time
from sklearn.utils import shuffle as skshuffle


parser = argparse.ArgumentParser(
    description='Distributed training for TransE'
)

parser.add_argument('--dataset', default='wordnet', metavar='',
                    help='dataset to be used: {wordnet, fb15k} (default: wordnet)')
parser.add_argument('--k', type=int, default=50, metavar='',
                    help='embedding dim (default: 50)')
parser.add_argument('--transe_gamma', type=float, default=1, metavar='',
                    help='TransE loss margin (default: 1)')
parser.add_argument('--transe_metric', default='l2', metavar='',
                    help='whether to use `l1` or `l2` metric for TransE (default: l2)')
parser.add_argument('--mbsize', type=int, default=100, metavar='',
                    help='size of minibatch (default: 100)')
parser.add_argument('--negative_samples', type=int, default=1, metavar='',
                    help='number of negative samples per positive sample  (default: 1)')
parser.add_argument('--nepoch', type=int, default=5, metavar='',
                    help='number of training epoch (default: 5)')
parser.add_argument('--lr', type=float, default=0.1, metavar='',
                    help='learning rate (default: 0.1)')
parser.add_argument('--lr_decay_every', type=int, default=10, metavar='',
                    help='decaying learning rate every n epoch (default: 10)')
parser.add_argument('--log_interval', type=int, default=100, metavar='',
                    help='interval between training status logs (default: 100)')
parser.add_argument('--checkpoint_dir', default='models/', metavar='',
                    help='directory to save model checkpoint, saved every epoch (default: models/)')
parser.add_argument('--use_gpu', default=False, type=bool, metavar='',
                    help='whether to run in the GPU or CPU (default: False <i.e. CPU>)')
parser.add_argument('--randseed', default=9999, type=int, metavar='',
                    help='resume the training from latest checkpoint (default: False')

args = parser.parse_args()


# Set random seed
np.random.seed(args.randseed)
mx.random.seed(args.randseed)

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

C = args.negative_samples

# Initialize model
model = TransE(n_e, n_r, args.k, args.transe_metric, args.use_gpu)

# Training params
n_epoch = args.nepoch
mb_size = args.mbsize  # 2x with negative sampling
print_every = args.log_interval
checkpoint_dir = '{}/{}'.format(args.checkpoint_dir.rstrip('/'), args.dataset)
checkpoint_path = '{}/{}.bin'.format(checkpoint_dir, 'transe')

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Trainer
scheduler = mx.lr_scheduler.FactorScheduler(args.lr_decay_every, factor=0.5)
optimizer = mx.optimizer.Adam(learning_rate=args.lr, lr_scheduler=scheduler)
solver = gluon.Trainer(model.collect_params(), optimizer)

# Begin training
for epoch in range(n_epoch):
    print('Epoch-{}'.format(epoch+1))
    print('----------------')

    it = 0

    # Shuffle and chunk data into minibatches
    mb_iter = get_minibatches(X_train, mb_size, shuffle=True)

    for X_mb in mb_iter:
        start = time()

        # Build batch with negative sampling
        m = X_mb.shape[0]

        # C x M negative samples
        X_neg_mb = np.vstack([sample_negatives(X_mb, n_e) for _ in range(C)])

        # Total training data: M + CM
        X_train_mb = np.vstack([X_mb, X_neg_mb])

        # Training step
        with autograd.record():
            y = model(X_train_mb)
            y_pos, y_neg = y[:m], y[m:]
            loss = model.ranking_loss(y_pos, y_neg, args.transe_gamma, C)
            loss.backward()

        solver.step(batch_size=C*m)
        model.normalize_embeddings()

        end = time()

        # Training logs
        if it % print_every == 0:
            mrr, hits10 = eval_embeddings(model, X_val_pos, n_e, k=10)

            # For TransE, show loss, mrr & hits@10
            print('Iter-{}; loss: {:.4f}; val_mrr: {:.4f}; val_hits@10: {:.4f}; time per batch: {:.2f}s'
                    .format(it, loss.asscalar(), mrr, hits10, end-start))

        it += 1

    print()

    # Checkpoint every epoch
    model.collect_params().save(checkpoint_path)
