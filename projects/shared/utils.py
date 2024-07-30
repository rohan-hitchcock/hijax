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


class ExponentialMovingAverage:
    def __init__(self, decay=0.95):
        self.decay = decay
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.decay * self.value + (1 - self.decay) * new_value

    def __float__(self):
        return float(self.value) if self.value is not None else 0.0

    def __format__(self, format_spec):
        return f"{float(self):{format_spec}}"
