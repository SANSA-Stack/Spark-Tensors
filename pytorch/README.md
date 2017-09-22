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
python run_baselines.py --model {rescal,ermlp,distmult,transe} --dataset {wordnet,fb15k}
```
See `python run_baselines.py --help` for further options, e.g. hyperparameters.

### Testing
```
python test_baselines.py --model {rescal,ermlp,distmult,transe} --dataset {wordnet,fb15k}
```
See `python test_baselines.py --help` for further options, e.g. hyperparameters.

## Dependencies

1. Python 3.5+
2. PyTorch 0.2+
3. Numpy
4. Scikit-Learn
5. Pandas
