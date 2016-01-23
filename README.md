# Spark-Tensors
Temporary repository for implementing tensor factorization algorithms on Apache Spark

Currently I am working on the following 3 algorithms:

1. PARAFAC (parallel algorithms given here: [GigaTensor](https://www.cs.cmu.edu/~epapalex/papers/gigatensor_KDD2012.pdf), [U. Kang's PhD thesis](http://datalab.snu.ac.kr/~ukang/papers/KangThesis.pdf)) Also check the [HaTen2 paper](https://www.cs.cmu.edu/~epapalex/papers/haten2_icde2015.pdf) that apparently improves upon GigaTensor.
2. RESCAL [RESCAL paper](http://www.icml-2011.org/papers/438_icmlpaper.pdf), [M. Nickel's PhD thesis](http://edoc.ub.uni-muenchen.de/16056/1/Nickel_Maximilian.pdf) (Spark-based distributed algorithm will be designed for this)
3. HolE [Holographic Embeddings of Knowledge Graphs](http://arxiv.org/pdf/1510.04935v2) (Spark-based distributed algorithm will be designed for this)

This will be divided across the Spark-RDF (interface, I/O, storage) and Spark-Sem-ML (algorithm) repositories eventually.
