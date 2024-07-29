import jax
from dataclasses import dataclass 
from typing import List, Type, Callable

def register_model(weights: List[str] = [], hparams: List[str] = []) -> Callable[[Type], Type]:
    """ Use as a decorator to register a custom class as a model."""
    def decorator(cls: Type) -> Type:
        cls = jax.tree_util.register_dataclass(
            cls, 
            data_fields=weights, 
            meta_fields=hparams
        )
        cls = dataclass(cls)
        return cls

    return decorator
