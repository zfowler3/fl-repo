import torch
import numpy as np
import torch.nn as nn

__all__ = ["flatten_weights"]
def register_buffer(module: torch.nn.Module, name: str, value: torch.Tensor):
    """Add a buffer to module.

    Args:
        module: Module
        name: Buffer name. Supports complex module names like 'model.conv1.bias'.
        value: Buffer value
    """
    module_path, _, name = name.rpartition('.')
    mod = module.get_submodule(module_path)
    mod.register_buffer(name, value)


def get_buffer(module, target):
    """Get module buffer.

    Remove after pinning to a version
    where https://github.com/pytorch/pytorch/pull/61429 is included.
    Use module.get_buffer() instead.
    """
    module_path, _, buffer_name = target.rpartition('.')

    mod: torch.nn.Module = module.get_submodule(module_path)

    if not hasattr(mod, buffer_name):
        raise AttributeError(f'{mod._get_name()} has no attribute `{buffer_name}`')

    buffer: torch.Tensor = getattr(mod, buffer_name)

    if buffer_name not in mod._buffers:
        raise AttributeError('`' + buffer_name + '` is not a buffer')

    return buffer

def flatten_weights(model, numpy_output=True):
    """
    Flattens a PyTorch model. i.e., concat all parameters as a single, large vector.
    :param model: PyTorch model
    :param numpy_output: should the output vector be casted as numpy array?
    :return: the flattened (vectorized) model parameters either as Numpy array or Torch tensors
    """
    all_params = []
    for param in model.parameters():
        all_params.append(param.view(-1))
    all_params = torch.cat(all_params)
    if numpy_output:
        return all_params.cpu().detach().numpy()
    return all_params
