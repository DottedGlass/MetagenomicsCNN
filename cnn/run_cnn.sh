#!/bin/bash

#SBATCH --job-name=train_cnn
#SBATCH --time=2:0:0
#SBATCH --partition=gpuk80
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=20GB

# load modules
module restore mymodules

# train cnn
~/.conda/envs/metagenomics/bin/python train.py
