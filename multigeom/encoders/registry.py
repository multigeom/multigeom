"""
Registries of backbones and readouts, by name.

An encoder is saved with a description of its architecture (which backbone and readout, with which arguments) next to its
weights, so that `NeuralEncoder.load` can rebuild it. The description refers to classes by their registered name:

    @register_backbone("my_cnn")
    class MyCNN(nn.Module):
        out_channels, out_spatial = 32, 16
        def __init__(self, width: int = 32): ...

The decorator also records the arguments each instance was created with. Backbones need the attributes `out_channels`
and `out_spatial` (the shape [B, C, S, S] of their output), which the readout is sized from.
"""
from __future__ import annotations

import functools
import inspect

BACKBONES: dict[str, type] = {}
READOUTS: dict[str, type] = {}


def _register(registry: dict, kind: str, name: str):
    def decorator(cls):
        if name in registry and registry[name] is not cls:
            raise ValueError(f"a {kind} named {name!r} is already registered ({registry[name].__name__})")
        original_init = cls.__init__
        signature = inspect.signature(original_init)

        @functools.wraps(original_init)
        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if "_config_kwargs" not in self.__dict__:                       # the outermost class of a hierarchy wins
                bound = signature.bind(self, *args, **kwargs)
                bound.apply_defaults()
                arguments = dict(bound.arguments)
                arguments.pop("self", None)
                object.__setattr__(self, "_config_kwargs", arguments)

        cls.__init__ = init
        cls._registry_name = name
        registry[name] = cls
        return cls

    return decorator


def register_backbone(name: str):
    """Class decorator: make a backbone available under `name` (needed to save and reload an encoder that uses it)."""
    return _register(BACKBONES, "backbone", name)


def register_readout(name: str):
    """Class decorator: make a readout available under `name`."""
    return _register(READOUTS, "readout", name)


def describe(module, registry: dict, kind: str) -> dict:
    """`{"name": ..., "kwargs": {...}}` of a registered backbone or readout instance, for the saved architecture."""
    name = getattr(type(module), "_registry_name", None)
    if name is None or registry.get(name) is not type(module):
        raise ValueError(
            f"the {kind} {type(module).__name__} is not registered, so the encoder cannot describe its architecture. "
            f"Decorate the class with @register_{kind}('some_name') to save and reload the encoder, or save its "
            f"state_dict yourself."
        )
    return {"name": name, "kwargs": dict(getattr(module, "_config_kwargs", {}))}
