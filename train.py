from torch.utils.data import DataLoader
from data import HeartFailureDataset
from models import LinearRegression
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import torch
import ipdb


# custom collate_fn
def collate_fn(data):
    feats = []
    labels = []
    for e in data:
        feats.append(e["feat"])
        labels.append(e["label"])

    return torch.tensor(feats), torch.tensor(labels)


"""Get dataset"""
train_data = HeartFailureDataset(split="train")
test_data = HeartFailureDataset(split="test")
train_loader = DataLoader(
    train_data, batch_size=64, shuffle=True, collate_fn=collate_fn
)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True, collate_fn=collate_fn)

"""Get Train Configurations"""
in_dim = 18
out_dim = 2
total_epoch = 100
learning_rate = 1e-3
model = LinearRegression(in_dim=in_dim, out_dim=out_dim)
criterion = CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


"""Start Training"""
for epoch in range(total_epoch):
    for feat, label in train_loader:
        out = model(feat)
        # ipdb.set_trace()
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        print(loss, end="\r")

"""Evaluation"""
total = 0
correct = 0
for feat, label in test_loader:
    out = model(feat)
    pred = torch.argmax(out, dim=-1)
    total += len(pred)
    correct += torch.sum(pred == label).item()
print("\n")
print(f"Total: {total}, Correct: {correct}, Accuracy: {round(correct/total*100, 2)}")
