#!/bin/bash

# Install dependencies
pip install pvlib pandas matplotlib PyYAML tqdm timm mamba_ssm einops

# Run training
python train.py
