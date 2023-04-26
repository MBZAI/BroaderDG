import torch

x = torch.randn(3, 4)
x
indices = torch.tensor([0, 2])
torch.index_select(x, 0, indices)
torch.index_select(x, 1, indices)