import torch
from stochastic_round.stochastic_ops import copy_stochastic_cuda_, copy_stochastic_, add_stochastic_cuda_, add_stochastic_
from torch import Tensor


dim = 1024

# create tensors
torch.manual_seed(42)
source = torch.randn(dim, dim, device='cuda')
target = torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16)

# base example
copy_stochastic_cuda_(target, source, seed=42)
baseline = target.clone()

# ensure that the results are equal
copy_stochastic_cuda_(target, source, seed=42)
baseline2 = target.clone()

print(baseline - baseline2)
print("Results are equal (should be True):", torch.equal(baseline, baseline2))


# ensure that the results are different
copy_stochastic_cuda_(target, source, seed=123)
changed_seed_result = target.clone()
print("Result1 equals Result3 (should be False):", torch.equal(baseline, changed_seed_result))


# test non-fused version
copy_stochastic_(target, source, seed=42)
non_fused_result = target.clone()
diff = baseline - non_fused_result
diff_mean = baseline.float().mean() - non_fused_result.float().mean()
print (diff)
print(diff_mean)
print("Results should be of same distribution:", (diff_mean.abs() < 1e-4).item())
from stochastic_round.copy_stochastic import copy_stochastic_


# lets try again witth ones
source = torch.ones(dim, dim, device='cuda') + 0.0002

# base example
copy_stochastic_cuda_(target, source, seed=42)
baseline = target.clone()

#test on non-fused version
copy_stochastic_(target, source, seed=42)
non_fused_result = target.clone()
diff = baseline - non_fused_result
diff_mean = baseline.float().mean() - non_fused_result.float().mean()
diff1_with_orig = source - baseline
diff2_with_orig = source - non_fused_result
print (diff)
print(diff_mean)
print("Results should be of same distribution:", (diff_mean.abs() < 1e-4).item())


# test adding
target_cu = torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16)
target_reg = target_cu.clone()
source = torch.randn(dim, dim, device='cuda')
add_stochastic_cuda_(target_cu, source, seed=42)

# test non-fused version
add_stochastic_(target_reg, source, seed=42)

diff = target_cu - target_reg
diff_mean = target_cu.float().mean() - target_reg.float().mean()
print (diff)
print(diff_mean)
print("Results should be of same distribution:", (diff_mean.abs() < 1e-4).item())



