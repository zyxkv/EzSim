import sys
from functools import wraps

import torch

import ezsim

from .tensor import Tensor

_torch_ops = (
    torch.tensor,
    torch.asarray,
    torch.as_tensor,
    torch.as_strided,
    torch.from_numpy,
    torch.zeros,
    torch.zeros_like,
    torch.ones,
    torch.ones_like,
    torch.arange,
    torch.range,
    torch.linspace,
    torch.logspace,
    torch.eye,
    torch.empty,
    torch.empty_like,
    torch.empty_strided,
    torch.full,
    torch.full_like,
    torch.rand,
    torch.rand_like,
    torch.randn,
    torch.randn_like,
    torch.randint,
    torch.randint_like,
    torch.randperm,
)


def torch_op_wrapper(torch_op):
    @wraps(torch_op)
    def _wrapper(*args, dtype=None, requires_grad=False, scene=None, **kwargs):
        if "device" in kwargs:
            ezsim.raise_exception("Device selection not supported. All ezsim tensors are on GPU.")

        if not ezsim._initialized:
            ezsim.raise_exception("EzSim not initialized yet.")

        if torch_op is torch.from_numpy:
            torch_tensor = torch_op(*args)
        else:
            torch_tensor = torch_op(*args, **kwargs)

        return from_torch(torch_tensor, dtype, requires_grad, detach=True, scene=scene)

    _wrapper.__doc__ = (
        f"This method is the ezsim wrapper of `torch.{torch_op.__name__}`.\n\n"
        "------------------\n"
        f"{_wrapper.__doc__}"
    )

    return _wrapper


def from_torch(torch_tensor, dtype=None, requires_grad=False, detach=True, scene=None):
    """
    By default, detach is True, meaning that this function returns a new leaf tensor which is not connected to torch_tensor's computation gragh.
    """
    if dtype is None:
        dtype = torch_tensor.dtype
    if dtype in (float, torch.float32, torch.float64):
        dtype = ezsim.tc_float
    elif dtype in (int, torch.int32, torch.int64):
        dtype = ezsim.tc_int
    elif dtype in (bool, torch.bool):
        dtype = torch.bool
    else:
        ezsim.raise_exception(f"Unsupported dtype: {dtype}")

    if torch_tensor.requires_grad and (not detach) and (not requires_grad):
        ezsim.logger.warning(
            "The parent torch tensor requires grad and detach is set to False. Ignoring requires_grad=False."
        )
        requires_grad = True

    ezsim_tensor = Tensor(torch_tensor.to(device=ezsim.device, dtype=dtype), scene=scene).clone()

    if detach:
        ezsim_tensor = ezsim_tensor.detach(sceneless=False)

    if requires_grad:
        ezsim_tensor = ezsim_tensor.requires_grad_()

    return ezsim_tensor


for _torch_op in _torch_ops:
    setattr(sys.modules[__name__], _torch_op.__name__, torch_op_wrapper(_torch_op))
