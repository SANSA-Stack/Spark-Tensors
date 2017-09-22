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

# Begin training
for epoch in range(n_epoch):
    print('Epoch-{}'.format(epoch+1))
    print('----------------')

    mb_iter = get_minibatches(X_train, mb_size)
    it = 0

    for X_mb in mb_iter:
        start = time()

        # Build batch with negative sampling and literals
        X_neg_mb = sample_negatives(X_mb, n_e=n_e)
        X_train_mb = np.hstack([X_mb, X_neg_mb])

        m = X_mb.shape[1]
        y_true_mb = np.vstack([np.ones([m, 1]), np.zeros([m, 1])])

        # Training step
        y = model.forward(X_train_mb)
        loss = model.loss(y, y_true_mb)
        loss.backward()
        solver.step()

        end = time()

        if it % print_every == 0:
            # Test on validation
            X_neg_val = sample_negatives(X_val, n_e=n_e)
            X_all_val = np.hstack([X_val, X_neg_val])
            y_true_val = np.vstack([np.ones([M_val, 1]), np.zeros([M_val, 1])])

            y_pred_val = model.predict(X_all_val)

            if args.model != 'transe':
                # Metrics
                val_acc = accuracy(y_pred_val, y_true_val)
                val_auc = auc(y_pred_val, y_true_val)

                print('Iter-{}; loss: {:.4f}; val_acc: {:.4f}; val_auc: {:.4f}; time per batch: {:.2f}s'
                      .format(it, loss.data[0], val_acc, val_auc, end-start))
            else:
                # For TransE, just show energy/loss
                print('Iter-{}; loss: {:.4f}; time per batch: {:.2f}s'
                      .format(it, loss.data[0], end-start))

        it += 1

    print()

    # Checkpoint every epoch
    torch.save(model.state_dict(), checkpoint_path)
