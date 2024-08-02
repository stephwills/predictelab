# EGNN for elaboratability prediction

Contains code for training an EGNN to predict the elaboratability of a compound.
The model is trained using synthetic data generated using AiZynthFinder.
The model is trained using PyTorch and PyTorchGeometric.

This repo is under development!

## Install

To install the necessary packages:
```
conda create -n predictelab python=3.9 -y
conda activate predictelab
conda install -c conda-forge rdkit -y
conda install -c conda-forge numpy -y
conda install -c conda-forge sklearn -y
conda install -c conda-forge scipy -y
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install pyg -c pyg -y
pip install joblib tqdm wandb biopython
```