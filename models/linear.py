import torch


class LinearRegression(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(in_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, out_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # MLP with polynomial activation function
        out = self.linear1(x)
        out = self.linear2(out)
        out = self.linear3(out)
        return out
