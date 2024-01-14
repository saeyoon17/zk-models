import torch


class LinearRegression(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        out = self.linear(x)
        return out
