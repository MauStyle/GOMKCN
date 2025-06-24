# GOMKCN: Substructure-Learnable Graph Disentangled Representation based on Optimal Matching Kernel
This repository is the official PyTorch implementation of "Substructure-Learnable Graph Disentangled Representation based on Optimal Matching Kernel", Mao Wang, Tao Wu, Xingping Xian, Chao Wang, Lin Yuan, Canyixing Cui, Weina Niu [link] (https://arxiv.org/abs/2504.16360).

## Installation
Install PyTorch following the instuctions on the [official website] (https://pytorch.org/). The code has been tested over PyTorch 2.5.1 version.

Then install the other dependencies.
```
pip install -r requirements.txt
```
## Datasets
All the dataset are downloaded from
```bash
[PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html)
```
The data spilts are the same with [here](https://github.com/diningphil/gnn-comparison).
## Test Run
```bash
python lancher - ENZYMES.py
```
