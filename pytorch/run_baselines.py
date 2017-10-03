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
parser.add_argument('--transe_gamma', type=float, default=1, metavar='',
                    help='TransE loss margin (default: 1)')
parser.add_argument('--transe_metric', default='l2', metavar='',
                    help='whether to use `l1` or `l2` metric for TransE (default: l2)')
parser.add_argument('--mlp_h', type=int, default=100, metavar='',
                    help='size of ER-MLP hidden layer (default: 100)')
parser.add_argument('--mlp_dropout_p', type=float, default=0.5, metavar='',
                    help='Probability of dropping out neuron in dropout (default: 0.5)')
parser.add_argument('--mbsize', type=int, default=100, metavar='',
                    help='size of minibatch (default: 100)')
parser.add_argument('--nepoch', type=int, default=5, metavar='',
                    help='number of training epoch (default: 5)')
parser.add_argument('--lr', type=float, default=0.1, metavar='',
                    help='learning rate (default: 0.1)')
parser.add_argument('--lr_decay_every', type=int, default=10, metavar='',
                    help='decaying learning rate every n epoch (default: 10)')
parser.add_argument('--weight_decay', type=float, default=1e-4, metavar='',
                    help='L2 weight decay (default: 1e-4)')
parser.add_argument('--log_interval', type=int, default=100, metavar='',
                    help='interval between training status logs (default: 100)')
parser.add_argument('--checkpoint_dir', default='models/', metavar='',
                    help='directory to save model checkpoint, saved every epoch (default: models/)')
parser.add_argument('--resume', default=False, metavar='',
                    help='resume the training from latest checkpoint (default: False')

args = parser.parse_args()


# Load dataset
X_train, n_e, n_r = load_data_bin('data/NTN/{}/bin/train.npy'.format(args.dataset))
X_val, _, _ = load_data_bin('data/NTN/{}/bin/val.npy'.format(args.dataset))
y_val = np.load('data/NTN/{}/bin/y_val.npy'.format(args.dataset))

X_val_pos = X_val[:, y_val.ravel() == 1]  # Take only positive samples

M_train = X_train.shape[1]
M_val = X_val.shape[1]

# Initialize model
models = {
    'rescal': RESCAL(n_e=n_e, n_r=n_r, k=args.k),
    'distmult': DistMult(n_e=n_e, n_r=n_r, k=args.k),
    'ermlp': ERMLP(n_e=n_e, n_r=n_r, k=args.k, h_dim=args.mlp_h, p=args.mlp_dropout_p),
    'transe': TransE(n_e=n_e, n_r=n_r, k=args.k, gamma=args.transe_gamma, d=args.transe_metric)
}

model = models[args.model]

# Resume from latest checkpoint if specified
if args.resume:
    model.load_state_dict(
        torch.load('models/{}/{}.bin'.format(args.dataset, args.model))
    )

# Training params
solver = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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

    it = 0

    # Shuffle and chunk data into minibatches
    mb_iter = get_minibatches(X_train, mb_size, shuffle=True)

    # Anneal learning rate
    lr = args.lr * (0.1 ** (epoch // args.lr_decay_every))
    for param_group in solver.param_groups:
        param_group['lr'] = lr

    for X_mb in mb_iter:
        start = time()

        # Build batch with negative sampling
        m = X_mb.shape[1]

        X_neg_mb = sample_negatives(X_mb, n_e)

        X_train_mb = np.hstack([X_mb, X_neg_mb])
        y_true_mb = np.vstack([np.ones([m, 1]), np.zeros([m, 1])])

        # Training step
        y = model.forward(X_train_mb)
        loss = model.loss(y, y_true_mb)
        loss.backward()
        solver.step()
        solver.zero_grad()
        model.normalize_embeddings()

        end = time()

        # Training logs
        if it % print_every == 0:
            if args.model != 'transe':
                # Training accuracy
                pred = model.predict(X_train_mb)
                train_acc = accuracy(pred, y_true_mb)

                # Per class training accuracy
                pos_acc = accuracy(pred[:m], np.ones([m, 1]))
                neg_acc = accuracy(pred[m:], np.zeros([m, 1]))

                # Validation accuracy
                y_pred_val = model.forward(X_val)
                val_acc = accuracy(y_pred_val.data.numpy(), y_val)

                # Validation loss
                val_loss = model.loss(y_pred_val, y_val).data[0]

                print('Iter-{}; loss: {:.4f}; train_acc: {:.4f}; pos: {:.4f}; neg: {:.4f}; val_acc: {:.4f}; val_loss: {:.4f}; time per batch: {:.2f}s'
                      .format(it, loss.data[0], train_acc, pos_acc, neg_acc, val_acc, val_loss, end-start))
            else:
                mrr, hits10 = eval_embeddings(model, X_val_pos, n_e, k=10)

                # For TransE, show loss, mrr & hits@10
                print('Iter-{}; loss: {:.4f}; val_mrr: {:.4f}; val_hits@10: {:.4f}; time per batch: {:.2f}s'
                      .format(it, loss.data[0], mrr, hits10, end-start))

        it += 1

    print()

    # Checkpoint every epoch
    torch.save(model.state_dict(), checkpoint_path)
