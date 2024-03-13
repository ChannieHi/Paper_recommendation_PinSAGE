# PinSAGE
This is the PinSAGE package applied to the research paper recommendation system.
It was implemented based on the DGL library and modified in the PinSAGE example.

PinSAGE paper: https://arxiv.org/pdf/1806.01973.pdf <br>
DGL: https://docs.dgl.ai/# <br>
DGL PinSAGE example: https://github.com/dmlc/dgl/tree/master/examples/pytorch/pinsage <br>

## Requirements

- - dgl
- - dask
- - pandas
- - torch
- - torchtext
- - sklearn

## Training model
```
python model.py -d data.pkl -s model -k 500 --eval-epochs 100 --save-epochs 100 --num-epochs 500 --device 0 --hidden-dims 128 --batch-size 64 --batches-per-epoch 512
```

- d: Data Files
- s: The name of the model to
- k: top K count
- eval epochs: performance output epoch interval (0 = output X)
- save epochs: storage epoch interval (0 = storage X)
- - num epochs: epoch 횟수
- hidden dims: embedding dimension
- batch size: batch size
- - batches per epoch: iteration 횟수
