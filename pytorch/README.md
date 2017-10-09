# KGA

Knowledge Graph Analysis in PyTorch.

## List of Models

1. RESCAL
2. ER-MLP
3. TransE
4. DistMult

## Getting Started

1. Install miniconda <http://conda.pydata.org/miniconda.html>
2. Do `conda env create`
3. Enter the env `source activate kga`
4. Install [Pytorch](https://github.com/pytorch/pytorch#installation)

## Running the Models
### Training
```
python run_baselines.py --model {rescal,ermlp,distmult,transe} --dataset {wordnet,fb15k} --use_gpu {True,False}
```
See `python run_baselines.py --help` for further options, e.g. hyperparameters:

```
optional arguments:
  -h, --help          show this help message and exit
  --model             model to run: {rescal, distmult, ermlp, transe}
                      (default: rescal)
  --dataset           dataset to be used: {wordnet, fb15k} (default: wordnet)
  --k                 embedding dim (default: 50)
  --transe_gamma      TransE loss margin (default: 1)
  --transe_metric     whether to use `l1` or `l2` metric for TransE (default:
                      l2)
  --mlp_h             size of ER-MLP hidden layer (default: 100)
  --mlp_dropout_p     Probability of dropping out neuron in dropout (default:
                      0.5)
  --mbsize            size of minibatch (default: 100)
  --nepoch            number of training epoch (default: 5)
  --lr                learning rate (default: 0.1)
  --lr_decay_every    decaying learning rate every n epoch (default: 10)
  --weight_decay      L2 weight decay (default: 1e-4)
  --normalize_embed   whether to normalize embeddings to unit euclidean ball
                      (default: False)
  --log_interval      interval between training status logs (default: 100)
  --checkpoint_dir    directory to save model checkpoint, saved every epoch
                      (default: models/)
  --resume            resume the training from latest checkpoint (default:
                      False
  --use_gpu           whether to run in the GPU or CPU (default: False <i.e.
                      CPU>)
```

### Testing
**Note:** All hyperparameters of the model _must_ match those used during training.

```
python test_baselines.py --model {rescal,ermlp,distmult,transe} --dataset {wordnet,fb15k} --use_gpu {True,False}
```
See `python test_baselines.py --help` for further options, e.g. hyperparameters:

```
optional arguments:
  -h, --help        show this help message and exit
  --model           model to run: {rescal, distmult, ermlp, transe} (default:
                    rescal)
  --dataset         dataset to be used: {wordnet, fb15k} (default: wordnet)
  --k               embedding dim (default: 50)
  --transe_gamma    TransE loss margin (default: 1)
  --transe_metric   whether to use `l1` or `l2` metric for TransE (default:
                    l2)
  --mlp_h           size of ER-MLP hidden layer (default: 100)
  --mlp_dropout_p   Probability of dropping out neuron in dropout (default:
                    0.5)
  --hit_k           hit@k metrics (default: 10)
  --nn_n            number of entities/relations for nearest neighbours
                    (default: 5)
  --nn_k            k in k-nearest-neighbours (default: 5)
  --use_gpu         whether to run in the GPU or CPU (default: False <i.e.
                    CPU>)
```

## Dependencies

1. Python 3.5+
2. PyTorch 0.2+
3. Numpy
4. Scikit-Learn
5. Pandas
