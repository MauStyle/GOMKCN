# GOMKCN: Substructure-Learnable Graph Disentangled Representation via Optimal Matching Kernels

## Installation
Install the other dependencies.
```
pip install -r requirements.txt
```
## Datasets
All datasets will be automatically downloaded from the TUDataset platform via the provided source code.
For node classification tasks, the training, validation, and test sets are randomly split.
For graph classification tasks, the data spilts are the same with [here](https://github.com/diningphil/gnn-comparison).

## Test Run
- Node classification on Pubmed dataset
```bash
python run-Pubmed.py
```
- Graph classification on ENZYMES dataset
```bash
python run-ENZYMES.py
```

##Acknowledgments
Thanks to the authors of the academic papers on kerGNN, RWK+CN, and GKNN, as well as the developers of the software platforms PyTorch, PyG, TUDataset, and torch-linear-assignment. Their research and valuable tools have provided invaluable reference and assistance for the implementation of this project.
