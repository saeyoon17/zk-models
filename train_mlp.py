import ipdb
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from data import HeartFailureDataset
from models import MLP


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
    train_data = HeartFailureDataset(split="train")
    test_data = HeartFailureDataset(split="test")
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    """Get Train Configurations"""
    torch.manual_seed(17)
    in_dim = 18
    hidden_dim = 4
    out_dim = 2
    total_epoch = 50
    learning_rate = 1e-4
    model = MLP(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, hidden_layer=6)
    criterion = CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    """Start Training"""
    for epoch in range(total_epoch):
        for feat, label in train_loader:
            out = model(feat)
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
    print(
        f"Total: {total}, Correct: {correct}, Accuracy: {round(correct/total*100, 2)}"
    )

    """Checkpointing"""
    PATH = f"./data/mlp_l8_hidden{hidden_dim}_ckpt.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "accuracy": round(correct / total * 100, 2),
        },
        PATH,
    )
