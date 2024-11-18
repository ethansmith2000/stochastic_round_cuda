import torch
from copy_stochastic import copy_stochastic_

source = torch.randn(1000, 1000, device='cuda', dtype=torch.float32)
target = torch.empty_like(source)

copy_stochastic_(target, source)

print("Source Tensor:", source)
print("Target Tensor:", target)
