import json
import os
import time
from collections import defaultdict

import ezkl
import ipdb
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import HeartFailureDataset
from models import MLP, LinearRegression
from train_linear_regression import collate_fn


def test_perf(num_trials, result):
    for trial_idx in tqdm(range(num_trials)):
        total = 0
        correct = 0
        zk_total = 0
        zk_correct = 0
        for idx, (feat, label) in enumerate(test_loader):
            model.eval()
            torch_out = model(feat)
            input_size = feat.size()
            output_size = torch_out.size()

            # Calculate model accuracy
            pred = torch.argmax(torch_out, dim=-1)
            total += len(pred)
            correct += torch.sum(pred == label).item()

            # Export the model
            torch.onnx.export(
                model,  # model being run
                feat,  # model input (or a tuple for multiple inputs)
                model_path,  # where to save the model (can be a file or file-like object)
                export_params=True,  # store the trained parameter weights inside the model file
                opset_version=10,  # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names=["input"],  # the model's input names
                output_names=["output"],  # the model's output names
                dynamic_axes={
                    "input": {0: "batch_size"},  # variable length axes
                    "output": {0: "batch_size"},
                },
            )

            data_array = ((feat).detach().numpy()).reshape([-1]).tolist()
            data = dict(
                input_shapes=[feat.shape],
                input_data=[data_array],
                output_data=[((torch_out).detach().numpy()).reshape([-1]).tolist()],
            )
            json.dump(data, open(data_path, "w"))

            py_run_args = ezkl.PyRunArgs()
            py_run_args.input_visibility = "public"
            py_run_args.output_visibility = "public"
            py_run_args.param_visibility = "private"  # private by default
            py_run_args.variables = [("batch_size", feat.size(0))]

            st = time.time()
            res = ezkl.gen_settings(model_path, settings_path, py_run_args=py_run_args)
            result["setting_time"].append(time.time() - st)
            assert res == True

            # calibration
            st = time.time()
            ezkl.calibrate_settings(
                data_path,
                model_path,
                settings_path,
                "resources",
                max_logrows=12,
                scales=[2],
            )
            result["calibration_time"].append(time.time() - st)

            st = time.time()
            res = ezkl.compile_circuit(model_path, compiled_model_path, settings_path)
            result["compile_time"].append(time.time() - st)
            assert res == True

            # srs path
            st = time.time()
            res = ezkl.get_srs(settings_path)
            result["get_srs_time"].append(time.time() - st)

            st = time.time()
            res = ezkl.gen_witness(data_path, compiled_model_path, witness_path)
            result["witness_generation_time"].append(time.time() - st)
            assert os.path.isfile(witness_path)
            with open(witness_path, "r") as f:
                wtns = json.load(f)

            scaled_input = torch.tensor(
                [float(e) for e in wtns["pretty_elements"]["rescaled_inputs"][0]]
            ).reshape(input_size)
            scaled_output = torch.tensor(
                [float(e) for e in wtns["pretty_elements"]["rescaled_outputs"][0]]
            ).reshape(output_size)
            zk_pred = torch.argmax(scaled_output, dim=-1)
            zk_total += len(zk_pred)
            zk_correct += torch.sum(zk_pred == label).item()

            st = time.time()
            res = ezkl.setup(
                compiled_model_path,
                vk_path,
                pk_path,
            )
            result["setup_time"].append(time.time() - st)

            assert res == True
            assert os.path.isfile(vk_path)
            assert os.path.isfile(pk_path)
            assert os.path.isfile(settings_path)

            proof_path = os.path.join("ezkl_data/test.pf")

            st = time.time()
            res = ezkl.prove(
                witness_path,
                compiled_model_path,
                pk_path,
                proof_path,
                "single",
            )
            result["proof_generation_time"].append(time.time() - st)
            print(res)
            assert os.path.isfile(proof_path)

            # VERIFY IT
            st = time.time()
            res = ezkl.verify(
                proof_path,
                settings_path,
                vk_path,
            )
            result["verification_time"].append(time.time() - st)
            assert res == True
            print("verified")

        # log results
        result["num_params"].append(num_params)
        result["total"].append(total)
        result["correct"].append(correct)
        result["zk_total"].append(zk_total)
        result["zk_correct"].append(zk_correct)


if __name__ == "__main__":
    """Get checkpoints, test dataset"""
    result = defaultdict(lambda: [])
    PATH = "./data/mlp_ckpt.pt"
    ckpt = torch.load(PATH)
    in_dim = 18
    hidden_dim = 8
    out_dim = 2
    num_trials = 1
    batch_size = 16
    model = MLP(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    test_data = HeartFailureDataset(split="test")
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    """EZKL configurations"""
    model_path = os.path.join("ezkl_data/network.onnx")
    compiled_model_path = os.path.join("ezkl_data/network.compiled")
    pk_path = os.path.join("ezkl_data/test.pk")
    vk_path = os.path.join("ezkl_data/test.vk")
    settings_path = os.path.join("ezkl_data/settings.json")
    witness_path = os.path.join("ezkl_data/witness.json")
    data_path = os.path.join("ezkl_data/input.json")
    cal_path = os.path.join("ezkl_data/calibration.json")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    test_perf(num_trials, result)
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
