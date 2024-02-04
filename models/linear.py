import torch


class LinearRegression(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, hidden_layer):
        super(MLP, self).__init__()
        in_linear = torch.nn.Linear(in_dim, hidden_dim)
        out_linear = torch.nn.Linear(hidden_dim, out_dim)
        self.linear = []
        self.linear.append(in_linear)
        for _ in range(hidden_layer):
            self.linear.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.linear.append(out_linear)
        self.linear = torch.nn.ModuleList(self.linear)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # MLP with polynomial activation function
        for m in self.linear[:-1]:
            x = self.relu(m(x))
        out = self.linear[-1](x)
        return out
