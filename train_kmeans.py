import pickle

import numpy as np
import torch
from hummingbird.ml import convert
from sklearn.cluster import KMeans

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
    x_train, x_test, y_train, y_test = HeartFailureDataset(split="train").get_data()

    """Get KMeans clustering does not require training"""
    clr = KMeans(n_clusters=2, random_state=17)
    clr.fit(x_train)

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
    PATH = f"./data/kmeans.pkl"
    with open(PATH, "wb") as f:
        pickle.dump(clr, f)
    
