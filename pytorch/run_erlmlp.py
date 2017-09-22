from kga.models import *
from kga.metrics import *
from kga.util import *
import numpy as np
import torch.optim


# Toy data
data = np.array([
    (0, 0, 2),
    (2, 1, 5),
    (1, 2, 4),
    (3, 3, 1)
])

literals = {
    0: (0, 30, 123123143, 0.1, 77, 95, 0, 7),
    1: (0.8, 55, 234234334, 0, 77, 0, 2, 9),
    2: (0.15, 42, 0, 0.2, 77, 70, 12, 40),
    3: (0, 26, 783122343, 0, 77, 120, 3, 0),
    4: (0.66, 0, 0, 0, 57, 76, 0, 0),
    5: (0, 0, 443122343, 0, 0, 0, 1, 0)
}


def build_literals(X):
    return np.array([literals[x[0]] for x in data], dtype=np.float32)


def standardize(X_lit, mean, std):
    return (X_lit - mean) / (std + 1e-8)


X_lit_all = np.array([literals[k] for k in range(6)], dtype=np.float32)
mean = np.mean(X_lit_all, axis=0)
std = np.std(X_lit_all, axis=0)

n_iter = 20
X = data.T
X_lit = build_literals(X.T)
X_lit = standardize(X_lit, mean, std)

model = ERLMLP(n_e=6, n_l=4, n_a=8, k=2, l=4, h_dim=64)

solver = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)

for it in range(n_iter):
    # Build batch with negative sampling and literals
    X_neg = sample_negatives(X, n_e=6)
    X_lit_neg = build_literals(X_neg)
    X_lit_neg = standardize(X_lit_neg, mean, std)

    X_train = np.hstack([X, X_neg])
    X_lit_train = np.vstack([X_lit, X_lit_neg])
    y_true = np.vstack([np.ones([X.shape[1], 1]), np.zeros([X.shape[1], 1])])

    # Training step
    y = model.forward(X_train, X_lit_train)
    loss = model.loss(y, y_true)
    loss.backward()
    solver.step()

    print(loss.data[0])

# Build test set
X_neg = sample_negatives(X, n_e=6)
X_lit_neg = build_literals(X_neg)
X_lit_neg = standardize(X_lit_neg, mean, std)

X_test = np.hstack([X, X_neg])
X_lit_test = np.vstack([X_lit, X_lit_neg])
y_true = np.vstack([np.ones([X.shape[1], 1]), np.zeros([X.shape[1], 1])])

# Predict and show metrics
y_pred = model.predict(X_test, X_lit_test)

print()
print('Accuracy: {}'.format(accuracy(y_pred, y_true)))
print('AUC: {}'.format(auc(y_pred, y_true)))
