import torch
import copy_stochastic_cuda

def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    if target.data_ptr() == source.data_ptr():
        return
    copy_stochastic_cuda.copy_stochastic_cuda(target, source)
