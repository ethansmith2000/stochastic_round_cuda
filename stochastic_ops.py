# stochastic_ops.py

import torch
import stochastic_ops_cuda

def checks(target, source):
    assert target.dtype == torch.bfloat16
    assert source.dtype == torch.float32

def copy_stochastic_cuda_(target, source, seed=None):
    checks(target, source)
    if seed is None:
        seed = torch.seed()
    stochastic_ops_cuda.copy_stochastic(target, source, seed)

def add_stochastic_cuda_(target, source, seed=None):
    checks(target, source)
    if seed is None:
        seed = torch.seed()
    stochastic_ops_cuda.add_stochastic(target, source, seed)

def sub_stochastic_cuda_(target, source, seed=None):
    checks(target, source)
    if seed is None:
        seed = torch.seed()
    stochastic_ops_cuda.sub_stochastic(target, source, seed)

def mul_stochastic_cuda_(target, source, seed=None):
    checks(target, source)
    if seed is None:
        seed = torch.seed()
    stochastic_ops_cuda.mul_stochastic(target, source, seed)

def div_stochastic_cuda_(target, source, seed=None):
    checks(target, source)
    if seed is None:
        seed = torch.seed()
    stochastic_ops_cuda.div_stochastic(target, source, seed)

def lerp_stochastic_cuda_(target, source, weight, seed=None):
    checks(target, source)
    if seed is None:
        seed = torch.seed()
    stochastic_ops_cuda.lerp_stochastic(target, source, weight, seed)

def addcmul_stochastic_cuda_(target, tensor1, tensor2, value, seed=None):
    checks(target, tensor1)
    checks(target, tensor2)
    if seed is None:
        seed = torch.seed()
    stochastic_ops_cuda.addcmul_stochastic(target, tensor1, tensor2, value, seed)

def addcdiv_stochastic_cuda_(target, tensor1, tensor2, value, seed=None):
    checks(target, tensor1)
    checks(target, tensor2)
    if seed is None:
        seed = torch.seed()
    stochastic_ops_cuda.addcdiv_stochastic(target, tensor1, tensor2, value, seed)


############################################################################################################
# python naked versions
############################################################################################################

def copy_stochastic_(target: torch.Tensor, source: torch.Tensor, seed=None):
    # for some reason randint_like does not allow a generator but randint does?

    # create a random 16 bit integer
    # if seed is None:
    #     seed = torch.seed()
    # generator = torch.Generator()
    # generator.manual_seed(seed)
    result = torch.randint_like(
        source,
        dtype=torch.int32,
        low=0,
        high=(1 << 16),
        # generator=generator,
    )

    # add the random number to the lower 16 bit of the mantissa
    result.add_(source.view(dtype=torch.int32))

    # mask off the lower 16 bit of the mantissa
    result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

    # copy the higher 16 bit into the target tensor
    target.copy_(result.view(dtype=torch.float32))


def add_stochastic_(target: torch.Tensor, source: torch.Tensor, seed=None):
    target32 = target.float()
    source32 = source.float()
    result32 = target32 + source32
    copy_stochastic_(target, result32, seed)

def sub_stochastic_(target: torch.Tensor, source: torch.Tensor, seed=None):
    target32 = target.float()
    source32 = source.float()
    result32 = target32 - source32
    copy_stochastic_(target, result32, seed)

def mul_stochastic_(target: torch.Tensor, source: torch.Tensor, seed=None):
    target32 = target.float()
    source32 = source.float()
    result32 = target32 * source32
    copy_stochastic_(target, result32, seed)

def div_stochastic_(target: torch.Tensor, source: torch.Tensor, seed=None):
    target32 = target.float()
    source32 = source.float()
    result32 = target32 / source32
    copy_stochastic_(target, result32, seed)

def lerp_stochastic_(target: torch.Tensor, source: torch.Tensor, weight: float, seed=None):
    target32 = target.float()
    source32 = source.float()
    result32 = target32 + weight * (source32 - target32)
    copy_stochastic_(target, result32, seed)

def addcmul_stochastic_(target: torch.Tensor, tensor1: torch.Tensor, tensor2: torch.Tensor, value: float, seed=None):
    target32 = target.float()
    tensor132 = tensor1.float()
    tensor232 = tensor2.float()
    result32 = target32 + tensor132 * tensor232 * value
    copy_stochastic_(target, result32, seed)

def addcdiv_stochastic_(target: torch.Tensor, tensor1: torch.Tensor, tensor2: torch.Tensor, value: float, seed=None):
    target32 = target.float()
    tensor132 = tensor1.float()
    tensor232 = tensor2.float()
    result32 = target32 + tensor132 / tensor232 * value
    copy_stochastic_(target, result32, seed)


