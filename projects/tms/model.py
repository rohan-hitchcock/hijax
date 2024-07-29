import jax
import jax.numpy as jnp

from collections import namedtuple

from dataclasses import dataclass

from functools import partial

from jaxtyping import Array, PRNGKeyArray as Key

from shared.utils import register_model

@register_model(weights=['matrix', 'bias'])
class TMSModel:

    matrix: Array
    bias: Array

    def __call__(self, x):

        x_compressed = jnp.matmul(x, self.matrix.T)
        x_recovered = jax.nn.relu(jnp.matmul(x_compressed, self.matrix) + self.bias)

        return x_recovered
    
    def save(self, filepath):

        if not filepath.endswith('.npz'):
            filepath = filepath + '.npz'

        jnp.savez(filepath, matrix=self.matrix, bias=self.bias)

    @classmethod
    def initialize(cls, key: Key, in_dim, hidden_dim):

        bias = jnp.zeros(in_dim)
        bound = jnp.sqrt(1 / in_dim)
        matrix = jax.random.uniform(key, shape=(hidden_dim, in_dim), minval=-bound, maxval=bound)

        return cls(bias=bias, matrix=matrix)

    @classmethod
    def load(cls, filepath):
        data = jnp.load(filepath)
        model = cls(matrix=data['matrix'], bias=data['bias'])
        return model

def loss_fn(model: TMSModel, x: Array):
    x_pred = model(x)
    return jnp.mean((x - x_pred) ** 2)
