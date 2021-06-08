# DDP-GCN: Multi-Graph Convolutional Network for Spatiotemporal Traffic Forecasting

<img src="https://github.com/snu-adsl/DDP-GCN/blob/main/images/model-full.PNG" width="700" align="center">

<img src="https://github.com/snu-adsl/DDP-GCN/blob/main/images/model-spatial.PNG" width="600" align="center">

This is a TensorFlow implementation of DDP-GCN in the following paper: https://arxiv.org/abs/1905.12256.
Our codes are mostly built upon the codes of https://github.com/VeritasYin/STGCN_IJCAI-18.

## Requirements
Our code is based on Python3. There are a few dependencies to run the code. We list the major libraries as below.
* tensorflow >= 1.9.0
* numpy >= 1.15
* pandas >= 0.23
* scipy >= 1.1.0

Dependency can be installed using the following command:
```
pip install -r requirements.txt
```

## Dataset
We provide the raw dataset in https://github.com/snu-adsl/ddpgcn-dataset.
### Graph construction
We defined three types of weighted graphs based on the distances and the direction among the link vectors in the traffic network.
For each graph, adjacency matrix W can be computed as below.

<img src="https://github.com/snu-adsl/DDP-GCN/blob/main/images/graph-distance.PNG" width="500" align="center">

<img src="https://github.com/snu-adsl/DDP-GCN/blob/main/images/graph-direction.PNG" width="500" align="center">

<img src="https://github.com/snu-adsl/DDP-GCN/blob/main/images/graph-positional.PNG" width="500" align="center">

We provide the pre-defined weighted graphs in dataset directory.
Individual files in a repository are limited to a 100MB maximum size limit, we zip speed files. Please unzip the files when you use the speed data.

### Dataset pre-processing
Similar with the previous studies, we normalize the input data by z-score method.

## Model training
We provide the default hyperparamter setups for DDP-GCN(Single), DDP-GCN(Parallel), and DDP-GCN(Stacked) in model-Urban1.json.
The results are also available at output directory.

## Citation
If you need more detailed information about implementation, please read and cite the following paper:

```
@article{lee2019ddp,
  title={DDP-GCN: Multi-graph convolutional network for spatiotemporal traffic forecasting},
  author={Lee, Kyungeun and Rhee, Wonjong},
  journal={arXiv preprint arXiv:1905.12256},
  year={2019}
}
```
