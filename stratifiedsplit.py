#!/bin/python

import codecs
import sys
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit

with codecs.open("/data/yago/subgraph/exp/all.txt", encoding='utf-8') as t:
	def splitTriples(triple):
		s = triple.split()
		return np.array([[s[0], s[1], "_".join(s[2:])]])
	triples = [splitTriples(triple) for triple in t.readlines()]
	x = np.concatenate(triples)
	y = x[:,1]

sss = StratifiedShuffleSplit(y, n_iter=1, test_size=50000)
for train_index, test_index in sss:
	more = np.random.choice(X_train, 50000, replace=False)
	test_index = np.concatenate(test_index, more)
	np.delete(train_index, more)
	X_train, X_test = x[train_index], x[test_index]

with codecs.open("train.txt", "w", encoding='utf-8') as train:
	for i in X_train:
			train.write("%s\n" % "\t".join(i))

with codecs.open("test.txt", "w", encoding='utf-8') as test:
	for i in X_test:
			test.write("%s\n" % "\t".join(i))






# 0.023699999,0.040200002
# 0.041103683
# 0.023764705,0.040411763
# 0.04092149

# 0.52129996,0.84299994
# 0.31838977
# 0.52066666,0.8413333
# 0.3204099

# 0.46729994,0.8714
# 0.33456916
# 0.46233335,0.8705
# 0.32917634

# 2 epoch, 150, l2
# 0.022419998,0.059599996
# 0.9245831
# 0.023196077,0.05994118
# 0.9248461

# 10 epoch, 150, dot
# 0.0367,0.05218
# 0.9329041
# 0.03562745,0.050960794
# 0.93194866

# 10 epoch, 300, dot
# 0.0355,0.049539994
# 0.93533766
# 0.036803916,0.05070589
# 0.9338529

0.9991,0.9995001
0.9949927
0.9338235,0.9450393
0.9113489
