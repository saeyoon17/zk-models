import json
import os
import time
from collections import defaultdict
from subprocess import check_output

import ipdb
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import HeartFailureDataset
from models import LinearRegression, MLP
from train_linear_regression import collate_fn


def test_perf(num_trials, result):
    for trial_idx in tqdm(range(num_trials)):
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

            scaling = float(10**6)
            bias_scaling = float(10**12)
            feat_scaled = [int(e * scaling) for e in feat.reshape(-1).tolist()]
            linear_weight_scaled = [
                int(e * scaling) for e in linear_weight.T.reshape(-1).tolist()
            ]
            linear_bias_scaled = [
                int(e * bias_scaling) for e in linear_bias.reshape(-1).tolist()
            ]

            # proxy for performance degradation: this calculates the same value as circom circuit.
            a = (feat * scaling).to(torch.int64)
            b = torch.tensor(linear_weight_scaled).reshape(18, -1)
            c = torch.tensor(linear_bias_scaled)
            proxy = a @ b + c
            circom_pred = torch.argmax(proxy, dim=-1)
            zk_total += len(pred)
            zk_correct += torch.sum(pred == label).item()

            # PAD if batch size is smaller for circom proof
            if feat.size(0) != batch_size:
                feat = torch.cat(
                    [feat, torch.zeros((batch_size - feat.size(0), feat.size(1)))]
                )
            torch_out = model(feat)
            feat_scaled = [int(e * scaling) for e in feat.reshape(-1).tolist()]
            out = {"out": torch_out.reshape(-1).tolist()}
            # dump files
            with open(f"circom_data/output_{idx}.json", "w") as f:
                json.dump(out, f)

            input = {
                "a": feat_scaled,
                "b": linear_weight_scaled,
                "bias": linear_bias_scaled,
            }
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
                    f"node ./circom_circuits/mlp_js/generate_witness.js ./circom_circuits/mlp_js/mlp.wasm ./circom_data/input_{i}.json ./circom_data/witness_{i}.wtns"
                ],
                shell=True,
            )
            result["witness_generation_time"].append(time.time() - st)
            st = time.time()
            b = check_output(
                [
                    f"snarkjs groth16 prove ./circom_circuits/prove01.key ./circom_data/witness_{i}.wtns ./circom_data/proof_{i}.json ./circom_data/public_{i}.json"
                ],
                shell=True,
            )
            result["proof_generation_time"].append(time.time() - st)
            st = time.time()
            c = check_output(
                [
                    f"snarkjs groth16 verify ./circom_circuits/verification_key.json ./circom_data/public_{i}.json ./circom_data/proof_{i}.json"
                ],
                shell=True,
            )
            result["verification_time"].append(time.time() - st)

        result["num_params"].append(num_params)
        result["total"].append(total)
        result["correct"].append(correct)
        result["zk_total"].append(zk_total)
        result["zk_correct"].append(zk_correct)


if __name__ == "__main__":
    """Get checkpoints, test dataset"""
    PATH = "./data/mlp_ckpt.pt"
    ckpt = torch.load(PATH)
    result = defaultdict(lambda: [])
    in_dim = 18
    out_dim = 2
    num_trials = 1
    batch_size = 64
    model = MLP(in_dim=in_dim, out_dim=out_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    test_data = HeartFailureDataset(split="test")
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    ipdb.set_trace()
    linear1_weight = model.state_dict()["linear1.weight"]
    linear1_bias = model.state_dict()["linear1.bias"]
    linear2_weight = model.state_dict()["linear2.weight"]
    linear2_bias = model.state_dict()["linear2.bias"]
    linear3_weight = model.state_dict()["linear3.weight"]
    linear3_bias = model.state_dict()["linear3.bias"]
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    test_perf(num_trials, result)
    print("===== RESULT =====")
    print(f"===== {num_trials} TRIALS =====")
    for k, v in result.items():
        print(f"AVG {k}: {np.mean(v)}±{np.std(v)}")

    avg_acc = np.sum(result["correct"]) / np.sum(result["total"]) * 100
    zk_avg_acc = np.sum(result["zk_correct"]) / np.sum(result["zk_total"]) * 100
    print(f"AVG ACCURACY: {avg_acc}")
    print(f"ZK AVG ACCURACY: {zk_avg_acc}")
