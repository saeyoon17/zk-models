import json
import pickle
import time
from collections import defaultdict
from subprocess import check_output

import ipdb
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.cluster import KMeans

from data import HeartFailureDataset
from train_linear_regression import collate_fn


# num_trials, result, center, model
def test_perf(num_trials, result, center, model):
    for trial_idx in tqdm(range(num_trials)):
        labels = []
        preds = []
        total = 0
        correct = 0
        zk_total = 0
        zk_correct = 0
        scaling_factor = 1e3
        center = center * scaling_factor 
        for idx, (feat, label) in enumerate(test_loader):
            
            pred = torch.tensor(model.predict(feat))
            # Calculate model accuracy
            total += len(pred)
            # ipdb.set_trace()
            correct += torch.sum(pred == label).item()

            # scaling layers
            proxy_model = KMeans(n_clusters=2, init=center, n_init=1)
            proxy_model.fit(center)
            proxy_model.cluster_centers_ = center
            # ipdb.set_trace()

            circom_pred = torch.tensor(proxy_model.predict(feat*scaling_factor))
            labels = labels + label.tolist()
            preds = preds + circom_pred.tolist()
            zk_total += len(pred)
            zk_correct += torch.sum(circom_pred == label).item()

            # PAD if batch size is smaller for circom proof
            if feat.size(0) != batch_size:
                feat = torch.cat(
                    [feat, torch.zeros((batch_size - feat.size(0), feat.size(1)))]
                )

            # ipdb.set_trace()
            feat = [int(e * scaling_factor) for e in feat.reshape(-1).tolist()]
            e0 = [int(e) for e in center[0].reshape(-1).tolist()]
            e1 = [int(e) for e in center[1].reshape(-1).tolist()]
            circom_input = {'data': feat, 'e0': e0, 'e1': e1}
            # ipdb.set_trace()
            # dump files
            with open(f"circom_data/input_{idx}.json", "w") as f:
                json.dump(circom_input, f)
        accuracy = accuracy_score(labels, preds)
        prec = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        f1 = f1_score(labels, preds)
        print(accuracy, prec, recall, f1)

        # Proving
        for i in range(len(test_loader)):
            st = time.time()
            a = check_output(
                [
                    f"node ./circom_data/kmeans_js/generate_witness.js ./circom_data/kmeans_js/kmeans.wasm ./circom_data/input_{i}.json ./circom_data/witness_{i}.wtns"
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

        result["total"].append(total)
        result["correct"].append(correct)
        result["zk_total"].append(zk_total)
        result["zk_correct"].append(zk_correct)


if __name__ == "__main__":
    """Get checkpoints, test dataset"""
    PATH = f"./data/kmeans.pkl"
    with open(PATH, "rb") as f:
        model = pickle.load(f)
    center = model.cluster_centers_
    result = defaultdict(lambda: [])
    num_trials = 1
    batch_size = 16
    test_data = HeartFailureDataset(split="test")
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_perf(num_trials, result, center, model)
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
