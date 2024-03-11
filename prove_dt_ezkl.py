import json
import os
import pickle
import time
from collections import defaultdict

import ezkl
import ipdb
import numpy as np
import torch
from hummingbird.ml import convert
from sklearn import tree
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.tree import DecisionTreeClassifier as De
from sklearn.tree import _tree
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import HeartFailureDataset
from models import MLP, LinearRegression
from train_linear_regression import collate_fn


def test_perf(num_trials, result):
    labels = []
    preds = []
    total = 0
    correct = 0
    zk_total = 0
    zk_correct = 0
    for trial_idx in tqdm(range(num_trials)):
        feat = torch.tensor(np.array(x_test))
        label = torch.tensor(np.array(y_test))
        torch_out = torch.tensor(model.predict(x_test))
        input_size = np.array(feat).shape   
        output_size = (torch_out.shape[0], 1)
        circuit = convert(model, "torch", x_test[:1]).model
        # Export the model
        torch.onnx.export(
            circuit,  # model being run
            feat,  # model input (or a tuple for multiple inputs)
            model_path,  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=10,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=["input"],  # the model's input names
            output_names=["output"],  # the model's output names
            dynamic_axes={
                "input": {0: "batch_size"},  # variable length axes
                "output": {0: "batch_size"},  # variable length axes
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
            max_logrows=13,
            scales=[4],
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
            zk_pred = scaled_output.view(-1)
            zk_total += zk_pred.size(0)
            zk_correct += torch.sum(zk_pred == label).item()
            labels = labels + label.tolist()
            preds = preds + zk_pred.tolist()

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
            # print(res)
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
        result["total"].append(total)
        result["correct"].append(correct)
        result["zk_total"].append(zk_total)
        result["zk_correct"].append(zk_correct)
        accuracy = accuracy_score(labels, preds)
        prec = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        f1 = f1_score(labels, preds)
        print(accuracy, prec, recall, f1)


if __name__ == "__main__":
    """Get checkpoints, test dataset"""
    result = defaultdict(lambda: [])
    in_dim = 18
    hidden_dim = 4
    out_dim = 2
    num_trials = 1
    batch_size = 16
    PATH = f"./data/decision_tree.pkl"
    with open(PATH, "rb") as f:
        model = pickle.load(f)
    
    batch_size = 16
    x_train, x_test, y_train, y_test = HeartFailureDataset(split="train").get_data()

    """EZKL configurations"""
    model_path = os.path.join("ezkl_data/network.onnx")
    compiled_model_path = os.path.join("ezkl_data/network.compiled")
    pk_path = os.path.join("ezkl_data/test.pk")
    vk_path = os.path.join("ezkl_data/test.vk")
    settings_path = os.path.join("ezkl_data/settings.json")
    witness_path = os.path.join("ezkl_data/witness.json")
    data_path = os.path.join("ezkl_data/input.json")
    cal_path = os.path.join("ezkl_data/calibration.json")

    test_perf(num_trials, result)
    print("===== RESULT =====")
    print(f"===== {num_trials} TRIALS =====")
    for k, v in result.items():
        if "_time" in k:
            print(f"SUM {k}: {np.sum(v)}")
        else:
            print(f"AVG {k}: {np.mean(v)}±{np.std(v)}")

    total = len(y_test)
    correct = 0
    pred = model.predict(x_test)
    correct += np.sum(pred == y_test).item()
    avg_acc = correct / total * 100
    zk_avg_acc = np.sum(result["zk_correct"]) / np.sum(result["zk_total"]) * 100
    print(f"AVG ACCURACY: {avg_acc}")
    print(f"ZK AVG ACCURACY: {zk_avg_acc}")
