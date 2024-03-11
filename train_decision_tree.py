import pickle

import ipdb
import numpy as np
import torch
from hummingbird.ml import convert
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier as De
from sklearn.tree import _tree

from data import HeartFailureDataset


# custom collate_fn
def collate_fn(data):
    feats = []
    labels = []
    for e in data:
        feats.append(e["feat"])
        labels.append(e["label"])

    return torch.tensor(feats), torch.tensor(labels)


if __name__ == "__main__":
    """Get dataset"""
    batch_size = 16
    x_train, x_test, y_train, y_test = HeartFailureDataset(split="train").get_data()

    """Get Train Configurations"""
    # todo: fix scikit-learn seed
    # torch.manual_seed(17)
    clr = De(max_depth=3)
    clr.fit(x_train, y_train)

    circuit = convert(clr, "torch", x_test[:1]).model

    """Evaluation"""
    total = len(y_test)
    correct = 0
    pred = clr.predict(x_test)
    correct += np.sum(pred == y_test).item()
    print("\n")
    print(
        f"Total: {total}, Correct: {correct}, Accuracy: {round(correct/total*100, 2)}"
    )

    """Checkpointing"""
    PATH = f"./data/decision_tree.pkl"
    with open(PATH, "wb") as f:
        pickle.dump(clr, f)
