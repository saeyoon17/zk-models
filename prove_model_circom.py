"""Proving model using circom
1. Get linear layer weights
2. 
"""

import json
import os

import ezkl
import ipdb
import torch
from torch.utils.data import DataLoader

from data import HeartFailureDataset
from models import LinearRegression

from train import collate_fn

"""Get checkpoints, test dataset"""

PATH = "./model_ckpt.pt"
ckpt = torch.load(PATH)
in_dim = 18
out_dim = 2
scaling_factor = 1e9
model = LinearRegression(in_dim=in_dim, out_dim=out_dim)
model.load_state_dict(ckpt["model_state_dict"])

linear_weight = (
    model.state_dict()["linear.weight"].reshape(-1) * scaling_factor.tolist()
)
linear_bias = model.state_dict()["linear.bias"].reshape(-1) * scaling_factor.tolist()


ipdb.set_trace()
