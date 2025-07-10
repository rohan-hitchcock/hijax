from typing import List, Tuple
import itertools

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray as Key, Int, Float

from dln.dln_llc_constraint_problem import compute_delta_sigma
from shared.utils import register_model

import math


def xavier_uniform_init(key: Key, shape: Tuple[int, int]) -> Float[Array, "shape[0] shape[1]"]:
    bound = jnp.sqrt(6 / (shape[0] + shape[1]))
    return jax.random.uniform(key, shape, minval=-bound, maxval=bound)

def xavier_normal_init(key: Key, shape: Tuple[int, int]) -> Float[Array, "shape[0] shape[1]"]:
    std = jnp.sqrt(2 / (shape[0] + shape[1]))
    return std * jax.random.normal(key, shape)

@register_model(weights=['layers'])
class DeepLinearNetwork:

    layers: List[Array]

    """def __post_init__(self) -> None:
        
        for layer in self.layers:
            print(layer.shape)

        assert all(len(layer.shape) == 2 for layer in self.layers)
        assert all(
            layer_in.shape[1] == layer_out.shape[0] 
            for layer_in, layer_out in zip(self.layers, self.layers[1:])
        )"""

    def __call__(self, x: Array) -> Array:
        # x has shape [B, N] where N is the input dimension and B is the batch 
        # size, so our vectors are row vectors. jnp.linalg.multi_dot figures out 
        # the optimal order to several successive matrix multiplications
        # return jnp.linalg.multi_dot([x, *self.layers])
        result = x
        for layer in self.layers:
            result = result @ layer

        return result

    @staticmethod
    def loss(model, x: Array, y: Array) -> Array:
        y_pred = model(x)
        return jnp.mean((y - y_pred) ** 2)

    @jax.jit
    def rank(self) -> Int[Array, '1']:
        return jnp.linalg.matrix_rank(jnp.linalg.multi_dot(self.layers))

    @property 
    def layer_sizes(self) -> Int[Array, '{len(self.layers) + 1}']:
        sizes = [layer.shape[0] for layer in self.layers]
        return jnp.array(sizes + [self.layers[-1].shape[1]])

    @property
    def num_parameters(self) -> int:
        return sum(jnp.size(layer) for layer in self.layers)
    
    @property
    def dim_in(self) -> int:
        return self.layers[0].shape[0]
    
    @property
    def dim_out(self) -> int:
        return self.layers[-1].shape[1]

    @classmethod
    def initialize(cls, key: Key, layer_sizes: List[int]):

        assert len(layer_sizes) >= 3, "Must have at least 2 layers"

        shapes = zip(layer_sizes, layer_sizes[1:])
        key_init_layers = jax.random.split(key, num=len(layer_sizes) - 1)

        # layers = [jax.random.normal(k, shape) for k, shape in zip(key_init_layers, shapes)]
        # layers = [jax.random.truncated_normal(k, shape=shape, upper=1, lower=-1) for k, shape in zip(key_init_layers, shapes)]
        # layers = [jax.random.uniform(k, shape=shape, minval=-0.5, maxval=0.5) for k, shape in zip(key_init_layers, shapes)]
        layers = [xavier_normal_init(k, shape) for k, shape in zip(key_init_layers, shapes)]
        # layers = [xavier_uniform_init(k, shape) for k, shape in zip(key_init_layers, shapes)]

        return cls(layers=layers)
    
    @classmethod
    def initialize_true(
        cls, 
        key: Key, 
        min_layers: int, 
        max_layers: int, 
        min_layer_size: int, 
        max_layer_size: int, 
        reduce_layer_rank_prob: float = 0.5
    ):

        assert min_layers >= 2, "Must have at least two layers"
        assert min_layer_size >= 1

        key, key_num_layers = jax.random.split(key)
        num_layers = jax.random.randint(key_num_layers, shape=(1,), minval=min_layers, maxval=max_layers).item()

        key, key_layer_sizes = jax.random.split(key)
        layer_sizes = jax.random.randint(key_layer_sizes, shape=(num_layers + 1,), minval=min_layer_size, maxval=max_layer_size)

        key, key_init_layers = jax.random.split(key)
        model = cls.initialize(key_init_layers, layer_sizes)


        for i, layer in enumerate(model.layers):

            key, key_choose_layer = jax.random.split(key)
            key, key_choose_rank = jax.random.split(key)

            if jax.random.uniform(key_choose_layer) > reduce_layer_rank_prob:
                continue
            
            smallest_dim = 0 if layer.shape[0] <= layer.shape[1] else 1
            max_rank = layer.shape[smallest_dim]


            rank = jax.random.randint(key_choose_rank, shape=(1,), minval=0, maxval=max_rank).item()

            if smallest_dim == 0:
                model.layers[i] = layer.at[rank:, :].set(0.0)

            else:
                model.layers[i] = layer.at[:, rank:].set(0.0)

        return model


    def learning_coefficient(self) -> Float[Array, '1']:

        rank = self.rank()
        layer_sizes = self.layer_sizes

        delta = layer_sizes - rank

        delta_sigma = compute_delta_sigma(delta.tolist())
        delta_sigma_sum = sum(delta_sigma)
        ell = len(delta_sigma) - 1

        a = delta_sigma_sum - (jnp.ceil(delta_sigma_sum / ell) - 1) * ell
        
        llc = (
            0.5 * (-(rank ** 2) + rank * (layer_sizes[0] + layer_sizes[-1]))
            + a * (ell - a) / (4 * ell) 
            - ell * (ell - 1) / (4 * ell ** 2) * (delta_sigma_sum ** 2)
            + 0.5 * sum(d1 * d2 for d1, d2 in itertools.combinations(delta_sigma, r=2))
        )
        return llc



    
