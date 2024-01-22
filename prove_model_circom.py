"""Proving model using circom
1. Get linear layer weights
2. 
"""

import json
import os
import time
from subprocess import check_output
from collections import defaultdict

import ipdb
import torch
from torch.utils.data import DataLoader

from data import HeartFailureDataset
from models import LinearRegression
from train import collate_fn

"""Get checkpoints, test dataset"""

PATH = "./model_ckpt.pt"
ckpt = torch.load(PATH)
result = defaultdict(lambda: 0)
in_dim = 18
out_dim = 2
input_scaling_factor = 1e9
weight_scaling_factor = 1e9
batch_size = 64
model = LinearRegression(in_dim=in_dim, out_dim=out_dim)
model.load_state_dict(ckpt["model_state_dict"])
test_data = HeartFailureDataset(split="test")
test_loader = DataLoader(
    test_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)

linear_weight = model.state_dict()["linear.weight"]
linear_bias = model.state_dict()["linear.bias"]
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = 0
correct = 0
zk_total = 0
zk_correct = 0
for idx, (feat, label) in enumerate(test_loader):
    # Flips the neural net into inference mode
    model.eval()
    torch_out = model(feat)

    # Calculate model accuracy
    pred = torch.argmax(torch_out, dim=-1)
    total += len(pred)
    correct += torch.sum(pred == label).item()

    # PAD if batch size is smaller
    if feat.size(0) != batch_size:
        feat = torch.cat([feat, torch.zeros((batch_size - feat.size(0), feat.size(1)))])
    torch_out = model(feat)
    feat_scaled = (feat * input_scaling_factor).to(torch.int).reshape(-1).tolist()
    linear_weight_scaled = (
        (linear_weight * weight_scaling_factor).to(torch.int).reshape(-1).tolist()
    )
    linear_bias_scaled = (
        (linear_bias * weight_scaling_factor * input_scaling_factor)
        .to(torch.int)
        .reshape(-1)
        .tolist()
    )

    out = {"out": torch_out.reshape(-1).tolist()}
    # dump files
    with open(f"circom_data/output_{idx}.json", "w") as f:
        json.dump(out, f)

    input = {"a": feat_scaled, "b": linear_weight_scaled, "bias": linear_bias_scaled}
    # dump files
    with open(f"circom_data/input_{idx}.json", "w") as f:
        json.dump(input, f)

# Before this, you manually need to execute:
# circom ./circom_circuits/linear_regression.circom --r1cs --wasm --sym
# snarkjs powersoftau new bn128 12 pot12_0000.ptau -v
# snarkjs powersoftau contribute pot12_0000.ptau pot12_0001.ptau --name="First contribution" -v

# snarkjs powersoftau prepare phase2 pot12_0001.ptau pot12_final.ptau -v
# snarkjs groth16 setup linear_regression.r1cs pot12_final.ptau proof0.key
# snarkjs zkey contribute proof0.key proof01.key --name="your name" -v
# snarkjs zkey export verificationkey proof01.key verification_key.json


# Proving
for i in range(len(test_loader)):
    st = time.time()
    a = check_output(
        [
            f"node ./circom_circuits/linear_regression_js/generate_witness.js ./circom_circuits/linear_regression_js/linear_regression.wasm ./circom_data/input_{i}.json ./circom_data/witness_{i}.wtns"
        ],
        shell=True,
    )
    result["witness_generation_time"] += time.time() - st
    st = time.time()
    b = check_output(
        [
            f"snarkjs groth16 prove ./circom_circuits/prove01.key ./circom_data/witness_{i}.wtns ./circom_data/proof_{i}.json ./circom_data/public_{i}.json"
        ],
        shell=True,
    )
    result["proof_generation_time"] += time.time() - st
    st = time.time()
    c = check_output(
        [
            f"snarkjs groth16 verify ./circom_circuits/verification_key.json ./circom_data/public_{i}.json ./circom_data/proof_{i}.json"
        ],
        shell=True,
    )
    result["verification_time"] += time.time() - st

result["num_params"] = num_params
result["total"] = total
result["correct"] = correct
ipdb.set_trace()
# node ./circom_circuits/linear_regression_js/generate_witness.js ./circom_circuits/linear_regression_js/linear_regression.wasm ./circom_data/inputx.json witnessx.wtns
# snarkjs groth16 prove proof01.key witnessx.wtns proofx.json publicx.json
# snarkjs groth16 verify verification-key.json publicx.json proofx.json


# Proof generation
