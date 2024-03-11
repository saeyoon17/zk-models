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

    txt = tree.export_text(clr)

    def tree_to_code(tree, feature_names):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        nodes = []

        def recurse(node, depth, idx):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                nodes.append([idx, name, threshold])
                idx = recurse(tree_.children_left[node], depth + 1, idx + 1)
                nodes.append([idx + 1, name, threshold])
                idx = idx + 1
                idx = recurse(tree_.children_right[node], depth + 1, idx + 1)
            else:
                nodes.append([-1, idx, np.argmax(tree_.value[node][0])])

            return idx

        recurse(0, 1, 0)
        print(nodes)

    feature_names = [str(i) for i in range(18)]
    tree_to_code(clr, feature_names)

    """Checkpointing"""
    PATH = f"./data/decision_tree.pkl"
    with open(PATH, "wb") as f:
        pickle.dump(clr, f)
