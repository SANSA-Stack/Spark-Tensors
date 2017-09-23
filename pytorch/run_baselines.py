from kga.models import *
from kga.metrics import *
from kga.util import *
import numpy as np
import torch.optim
import argparse
import os
from time import time


parser = argparse.ArgumentParser(
    description='Train baselines methods: RESCAL, DistMult, ER-MLP, TransE'
)

parser.add_argument('--model', default='rescal', metavar='',
                    help='model to run: {rescal, distmult, ermlp, transe} (default: rescal)')
parser.add_argument('--dataset', default='wordnet', metavar='',
                    help='dataset to be used: {wordnet, fb15k} (default: wordnet)')
parser.add_argument('--k', type=int, default=50, metavar='',
                    help='embedding dim (default: 50)')
parser.add_argument('--gamma', type=float, default=1, metavar='',
                    help='TransE loss margin (default: 1)')
parser.add_argument('--lr', type=float, default=0.1, metavar='',
                    help='learning rate (default: 0.1)')
parser.add_argument('--decay', type=float, default=1e-4, metavar='',
                    help='L2 weight decay (default: 1e-4)')
parser.add_argument('--h', type=int, default=100, metavar='',
                    help='size of ER-MLP hidden layer (default: 100)')
parser.add_argument('--mbsize', type=int, default=100, metavar='',
                    help='size of minibatch (default: 100)')
parser.add_argument('--nepoch', type=int, default=5, metavar='',
                    help='number of training epoch (default: 5)')
parser.add_argument('--log_interval', type=int, default=100, metavar='',
                    help='interval between training status logs (default: 100)')
parser.add_argument('--checkpoint_dir', default='models/', metavar='',
                    help='directory to save model checkpoint, saved every epoch (default: models/)')

args = parser.parse_args()


# Load dataset
X_train, n_e, n_r = load_data_bin('data/{}/bin/train.npy'.format(args.dataset))
X_val, _, _ = load_data_bin('data/{}/bin/val.npy'.format(args.dataset))

M_train = X_train.shape[1]
M_val = X_val.shape[1]

# Initialize model
models = {
    'rescal': RESCAL(n_e=n_e, n_r=n_r, k=args.k),
    'distmult': DistMult(n_e=n_e, n_r=n_r, k=args.k),
    'ermlp': ERMLP(n_e=n_e, n_r=n_r, k=args.k, h_dim=args.h),
    'transe': TransE(n_e=n_e, n_r=n_r, k=args.k, gamma=args.gamma)
}

model = models[args.model]

# Training params
solver = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
n_epoch = args.nepoch
mb_size = args.mbsize  # 2x with negative sampling
print_every = args.log_interval
checkpoint_dir = '{}/{}'.format(args.checkpoint_dir.rstrip('/'), args.dataset)
checkpoint_path = '{}/{}.bin'.format(checkpoint_dir, args.model)

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Load up negative samples
X_train_neg = np.load('data/{}/bin/train_neg.npy'.format(args.dataset))
idxs = np.random.choice(np.arange(X_train_neg.shape[1]), size=M_train, replace=False)
X_train_neg = X_train_neg[:, idxs]

# Load up validation set
X_neg_val = np.load('data/{}/bin/val_neg.npy'.format(args.dataset))
idxs = np.random.choice(np.arange(X_neg_val.shape[1]), size=M_val, replace=False)
X_neg_val = X_neg_val[:, idxs]
X_all_val = np.hstack([X_val, X_neg_val])
y_true_val = np.vstack([np.ones([M_val, 1]), np.zeros([M_val, 1])])


# Begin training
for epoch in range(n_epoch):
    print('Epoch-{}'.format(epoch+1))
    print('----------------')

    X_train_neg = sample_negatives2(X_train_neg, n_e)
    idxs = np.random.choice(np.arange(X_train_neg.shape[1]), size=M_train, replace=False)
    X_train_neg = X_train_neg[:, idxs]

    mb_iter = get_minibatches(X_train, mb_size)
    mb_neg_iter = get_minibatches(X_train_neg, mb_size)

    it = 0

    for X_mb, X_neg_mb in zip(mb_iter, mb_neg_iter):
        start = time()

        # Build batch with negative sampling and literals
        X_train_mb = np.hstack([X_mb, X_neg_mb])

        m = X_mb.shape[1]
        y_true_mb = np.vstack([np.ones([m, 1]), np.zeros([m, 1])])

        # Training step
        y = model.forward(X_train_mb)
        loss = model.loss(y, y_true_mb)
        loss.backward()
        solver.step()
        model.normalize_embeddings()

        end = time()

        if it % print_every == 0:
            # Test on validation
            y_pred_val = model.predict(X_all_val)

            if args.model != 'transe':
                # Metrics
                train_acc = accuracy(model.predict(X_train_mb), y_true_mb)
                val_acc = accuracy(y_pred_val, y_true_val)
                val_auc = auc(y_pred_val, y_true_val)

                print('Iter-{}; loss: {:.4f}; train_acc: {:.4f}; val_acc: {:.4f}; val_auc: {:.4f}; time per batch: {:.2f}s'
                      .format(it, loss.data[0], train_acc, val_acc, val_auc, end-start))
            else:
                # For TransE, just show energy/loss
                print('Iter-{}; loss: {:.4f}; time per batch: {:.2f}s'
                      .format(it, loss.data[0], end-start))

        it += 1

    print()

    # Checkpoint every epoch
    torch.save(model.state_dict(), checkpoint_path)
