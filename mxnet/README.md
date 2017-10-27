# TransE

TransE embedding model in MXNet.

## Getting Started

1. Install miniconda <http://conda.pydata.org/miniconda.html>
2. Do `conda env create`
3. Enter the env `source activate kga`
4. Install [MXNet](https://mxnet.incubator.apache.org/get_started/install.html)

## Running the Models
### Training
Example:
```
python run_transe.py --k 20 --lr 0.1 --mbsize 500
```
See `python run_transe.py --help` for further options, e.g. hyperparameters:

```
optional arguments:
  -h, --help           show this help message and exit
  --dataset            dataset to be used: {wordnet, fb15k} (default: wordnet)
  --k                  embedding dim (default: 50)
  --transe_gamma       TransE loss margin (default: 1)
  --transe_metric      whether to use `l1` or `l2` metric for TransE (default:
                       l2)
  --mbsize             size of minibatch (default: 100)
  --negative_samples   number of negative samples per positive sample
                       (default: 1)
  --nepoch             number of training epoch (default: 5)
  --lr                 learning rate (default: 0.1)
  --lr_decay_every     decaying learning rate every n epoch (default: 10)
  --log_interval       interval between training status logs (default: 100)
  --checkpoint_dir     directory to save model checkpoint, saved every epoch
                       (default: models/)
  --use_gpu            whether to run in the GPU or CPU (default: False <i.e.
                       CPU>)
  --randseed           resume the training from latest checkpoint (default:
                       False
```

## Dependencies

1. Python 3.5+
2. MXNet 0.11.0
3. Numpy
4. Scikit-Learn
5. Pandas
