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
from models import MLP, LinearRegression
from train_linear_regression import collate_fn


def test_perf(num_trials, result, ckpt, hidden_layer):
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

            # scaling layers
            scaling = float(10 ** 4)
            bias_scaling = scaling ** 2
            feat_scaled = [int(e * scaling) for e in feat.reshape(-1).tolist()]
            scaled_weights = dict()
            # ipdb.set_trace()
            for layer_idx in range(hidden_layer+2):
                linear_weight = ckpt[f'linear.{layer_idx}.weight']
                bias = ckpt[f'linear.{layer_idx}.bias']

                linear_weight_scaled = [
                    int(e * scaling) for e in linear_weight.T.reshape(-1).tolist()
                ]
                bias_scaled = [
                    int(e * bias_scaling*(scaling**(layer_idx))) for e in bias.reshape(-1).tolist()
                ]
                scaled_weights[f'weight{layer_idx+1}'] = linear_weight_scaled
                scaled_weights[f'bias{layer_idx+1}'] = bias_scaled

            # proxy for performance degradation: this calculates the same value as circom circuit.
            m = torch.nn.ReLU()
            proxy = (feat * scaling).to(torch.int64)
            for layer_idx in range(hidden_layer+2):
                if layer_idx == 0:
                    dim1 = in_dim
                    dim2 = hidden_dim
                elif layer_idx == hidden_layer+1:
                    dim1 = hidden_dim
                    dim2 = out_dim
                else:
                    dim1 = hidden_dim
                    dim2 = hidden_dim
                # ipdb.set_trace()
                linear_weight = torch.tensor(scaled_weights[f'weight{layer_idx+1}'], dtype=torch.int64).reshape(dim1, dim2)
                bias = torch.tensor(scaled_weights[f'bias{layer_idx+1}'], dtype=torch.int64)
                if layer_idx != hidden_layer +1:
                    proxy = m(proxy@linear_weight + bias)
                else:
                    proxy = proxy@linear_weight + bias
                # ipdb.set_trace()

            circom_pred = torch.argmax(proxy, dim=-1)
            zk_total += len(pred)
            zk_correct += torch.sum(circom_pred == label).item()

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

            scaled_weights['batch_in'] = feat_scaled
            # dump files
            with open(f"circom_data/input_{idx}.json", "w") as f:
                json.dump(scaled_weights, f)
        # ipdb.set_trace()

        # Before this, you manually need to execute:
        # cd circom_data
        # circom ../circom_circuits/mlp.circom --r1cs --wasm --sym
        # snarkjs powersoftau new bn128 19 pot19_0000.ptau -v
        # snarkjs powersoftau contribute pot19_0000.ptau pot19_0001.ptau --name="First contribution" -v

        # snarkjs powersoftau prepare phase2 pot19_0001.ptau pot19_final.ptau -v
        # snarkjs groth16 setup mlp.r1cs pot19_final.ptau proof0.key
        # snarkjs zkey contribute proof0.key proof01.key --name="your name" -v
        # snarkjs zkey export verificationkey proof01.key verification_key.json

        # Proving
        for i in range(len(test_loader)):
            st = time.time()
            a = check_output(
                [
                    f"node ./circom_data/mlp_l3_d4_js/generate_witness.js ./circom_data/mlp_l3_d4_js/mlp_l3_d4.wasm ./circom_data/input_{i}.json ./circom_data/witness_{i}.wtns"
                ],
                shell=True,
            )
            result["witness_generation_time"].append(time.time() - st)
            st = time.time()
            b = check_output(
                [
                    f"snarkjs groth16 prove ./circom_data/proof01.key ./circom_data/witness_{i}.wtns ./circom_data/proof_{i}.json ./circom_data/public_{i}.json"
                ],
                shell=True,
            )
            result["proof_generation_time"].append(time.time() - st)
            st = time.time()
            c = check_output(
                [
                    f"snarkjs groth16 verify ./circom_data/verification_key.json ./circom_data/public_{i}.json ./circom_data/proof_{i}.json"
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
    PATH = "./data/mlp_l3_hidden4_ckpt.pt"
    ckpt = torch.load(PATH)
    result = defaultdict(lambda: [])
    in_dim = 18
    hidden_dim = 4
    out_dim = 2
    num_trials = 1
    batch_size = 16
    hidden_layer = 1
    model = MLP(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, hidden_layer=hidden_layer)
    model.load_state_dict(ckpt["model_state_dict"])
    test_data = HeartFailureDataset(split="test")
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    # ipdb.set_trace()
    # weights = dict()
    # for layer_idx in range(1, hidden_layer+1):
    # linear1_weight = model.state_dict()["linear1.weight"]
    # linear1_bias = model.state_dict()["linear1.bias"]
    # linear2_weight = model.state_dict()["linear2.weight"]
    # linear2_bias = model.state_dict()["linear2.bias"]
    # linear3_weight = model.state_dict()["linear3.weight"]
    # linear3_bias = model.state_dict()["linear3.bias"]
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    test_perf(num_trials, result, ckpt['model_state_dict'], hidden_layer)
    print("===== RESULT =====")
    print(f"===== {num_trials} TRIALS =====")
    for k, v in result.items():
        if "_time" in k:
            print(f"SUM {k}: {np.sum(v)}")
        else:
            print(f"AVG {k}: {np.mean(v)}±{np.std(v)}")

    avg_acc = np.sum(result["correct"]) / np.sum(result["total"]) * 100
    zk_avg_acc = np.sum(result["zk_correct"]) / np.sum(result["zk_total"]) * 100
    print(f"AVG ACCURACY: {avg_acc}")
    print(f"ZK AVG ACCURACY: {zk_avg_acc}")
