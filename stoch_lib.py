import torch
from torch import Tensor
from typing import List
from torch.utils._pytree import tree_map
from torch.distributed.tensor import DTensor


def promote(x):
    if x in (torch.bfloat16, torch.float16):
        return torch.float32
    if hasattr(x, 'dtype') and x.dtype in (torch.bfloat16, torch.float16):
        return x.float()
    return x

def check_bf16(x):
    is_bf16 = False
    if not isinstance(x, torch.Tensor):
        # assume tuple, list
        is_bf16 = x[0].dtype == torch.bfloat16
    else:
        is_bf16 = x.dtype == torch.bfloat16
    return is_bf16


# @torch.compile(mode='max-autotune-no-cudagraphs', fullgraph=True)
def copy_stochastic_(target: Tensor, source: Tensor):
    """
    copies source into target using stochastic rounding

    Args:
        target: the target tensor with dtype=bfloat16
        source: the target tensor with dtype=float32
    """
    #TODO we reeeeealy shouldnt need this, way too much overhead, will await torch support
    if isinstance(target, DTensor):
        target_for_op = target.to_local()
    else:
        target_for_op = target
    if isinstance(source, DTensor):
        source_for_op = source.to_local()
    else:
        source_for_op = source
    # source_for_op = source
    # target_for_op = target

    # create a random 16 bit integer
    result = torch.randint_like(
        source_for_op,
        dtype=torch.int32,
        low=0,
        high=(1 << 16),
    )

    # add the random number to the lower 16 bit of the mantissa
    result.add_(source_for_op.view(dtype=torch.int32))

    # mask off the lower 16 bit of the mantissa
    result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

    # copy the higher 16 bit into the target tensor
    target_for_op.copy_(result.view(dtype=torch.float32))

    if isinstance(target, DTensor):
        target_for_op = DTensor.from_local(target_for_op, device_mesh=target.device_mesh, placements=target.placements, shape=target.shape, stride=target.stride())
        target.copy_(target_for_op)
    #     del target_for_op
    # if isinstance(source, DTensor):
    #     del source_for_op

    del result


def cast(x, dtype):
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return x


def stoch_op_(fn, *args, **kwargs):
    fn = getattr(torch, fn)
    main = promote(args[0])
    others = tree_map(lambda x: promote(x), args[1:])
    fn(main, *others, out=main, **kwargs)
    copy_stochastic_(args[0], main)

    # if isinstance(x, torch.Tensor):
    #     x32 = promote(x)
    #     getattr(x32, fn)(promote(y), promote(z), **kwargs)
    #     copy_stochastic_(x, x32)
    # else:
    #     x32 = [promote(a) for a in x]
    #     y32 = [promote(a) for a in y]
    #     # check if z[0] and y[0] are the same to avoid an extra cast
    #     if z[0]._cdata == y[0]._cdata:
    #         for_each_fn(x32, y32, y32, **kwargs)
    #     else:
    #         z32 = [promote(a) for a in z]
    #         for_each_fn(x32, y32, z32, **kwargs)
    #     copy_stochastic_list_(x, x32)

def det_op_(fn, *args, **kwargs):
    # make sure you are not creating new tensors for the first arg!
    fn = getattr(torch, fn)
    # for_each_fn = getattr(torch, "_foreach_" + fn + "_")

    # cast to whatever the first arg is
    others = tree_map(lambda x: cast(x, args[0].dtype), args[1:])
    fn(*args, *others, out=args[0], **kwargs)


# stochastic ops
def mul_(x, y, stoch=False):
    if stoch and check_bf16(x):
        stoch_op_('mul', x, y)
    else:
        det_op_('mul', x, y)


def div_(x, y, stoch=False):
    if stoch and check_bf16(x):
        stoch_op_('div', x, y)
    else:
        det_op_('div', x, y)

def lerp_(x, y, weight=1, stoch=False):
    if stoch and check_bf16(x):
        stoch_op_('lerp', x, y, weight=weight)
    else:
        det_op_('lerp', x, y, weight=weight)

def addcmul_(x, y, z, value=1, stoch=False):
    if stoch and check_bf16(x):
        stoch_op_('addcmul', x, y, z, value=value)
    else:
        det_op_('addcmul', x, y, z, value=value)

def addcdiv_(x, y, z, value=1, stoch=False):
    if stoch and check_bf16(x):
        stoch_op_('addcdiv', x, y, z, value=value)
    else:
        det_op_('addcdiv', x, y, z, value=value)

def add_(x, y, alpha=1, stoch=False):
    if stoch and check_bf16(x):
        stoch_op_('add', x, y, alpha=alpha)
    else:
        det_op_('add', x, y, alpha=alpha)



def compose_eval(ops_and_args, stoch=False):
    # ops_and_args = {
    #     "mul": [("grad", "exp_avg", "beta1"), {}],
    # }
    expr = ""
    for k, v in ops_and_args.items():
        expr += "torch." + k + "("
        # place all the args
        for arg in v[0]:
            expr += arg + ", "
        # add out=arg[0]
        expr += "out=" + v[0][0] + ", "
        # add any kwargs
        for k, v in v[1].items():
            expr += k + "=" + v + ", "
        expr += ")"
        expr += "\n"
    
    return expr
        


def compose(ops_and_args, args_dict, stoch=False):
    # ops_and_args = {
    #     "mul": [("grad", "exp_avg", "beta1"), {}],
    # }

    # whether first arg of the first op is bf16 will determine whether whole composed function is done with
    # our stochasting rounding or not
    ops_and_args_list = list(ops_and_args.values())
    if stoch and check_bf16(args_dict[ops_and_args_list[0][0][0]]):
        originals = [args_dict[ops_and_args_list[i][0][0]] for i in range(len(ops_and_args))]
        new_args_dict = {k: promote(v) for k, v in args_dict.items()}
    else:
        new_args_dict = args_dict 

    for k, v in ops_and_args.items():
        args_strings = v[0]
        kwargs_strings = {}
        if len(v) > 1:
            kwargs_strings = v[1]

        args = [new_args_dict[arg] for arg in args_strings]
        # be careful here, we're assuming that kwargs are either in the args_dict or are floats expressed as strings
        kwargs = {k: new_args_dict.get(v, float(v)) for k, v in kwargs_strings.items()}

        fn = getattr(torch, k)
        fn(*args, out=args[0], **kwargs)

    if stoch and check_bf16(args_dict[ops_and_args_list[0][0][0]]):
        # copy back to originals
        for i in range(len(ops_and_args)):
            copy_stochastic_(originals[i], new_args_dict[ops_and_args_list[i][0][0]])


if __name__ == "__main__":
    ops_and_args = {
        "mul": [("grad", "exp_avg"), {}],
        "add": [("exp_avg", "grad"), {"alpha": "1.0"}],
    }

    grad = torch.ones(10, 10)
    exp_avg = torch.ones(10, 10) * 2

    args_dict = {
        "grad": grad,
        "exp_avg": exp_avg,
    }

    compose(ops_and_args, args_dict, stoch=False)



    ops_and_args = {
        "mul": [("grad", "exp_avg"), {}],
        "add": [("exp_avg", "grad"), {"alpha": "1.0"}],
    }

    grad = torch.ones(10, 10).to(torch.bfloat16)
    exp_avg = torch.ones(10, 10).to(torch.bfloat16) * 2

    args_dict = {
        "grad": grad,
        "exp_avg": exp_avg,
    }

    compose(ops_and_args, args_dict, stoch=True)







