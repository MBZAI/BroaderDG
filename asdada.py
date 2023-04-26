import torch
import torch.nn.functional as F

focal_loss = torch.hub.load(
	'adeelh/pytorch-multi-class-focal-loss',
	model='FocalLoss',
	alpha=torch.tensor([.75, .25]),
	gamma=2,
	reduction='mean',
	force_reload=False
)

x, y = torch.randn(10, 2), (torch.rand(10) > .5).long()

loss = focal_loss(x, y)

print(loss)

focal_loss = torch.hub.load(
	'adeelh/pytorch-multi-class-focal-loss',
	model='FocalLoss',
	alpha=None,
	gamma=2,
	reduction='mean',
	force_reload=False
)

loss = focal_loss(x, y)

print(loss)

print(F.cross_entropy(x, y))