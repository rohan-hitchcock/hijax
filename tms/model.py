import jax
import jax.numpy as jnp

from collections import namedtuple

TMSWeights = namedtuple('TMSWeights', ['matrix', 'bias'])

def init_weights(key, in_dim, hidden_dim):

    bias = jnp.zeros(in_dim)

    bound = jnp.sqrt(1 / in_dim)
    matrix = jax.random.uniform(key, shape=(hidden_dim, in_dim), minval=-bound, maxval=bound)

    return TMSWeights(matrix, bias)

def forward_pass(tms_weights, x):

    x_compressed = jnp.matmul(x, tms_weights.matrix.T)
    x_recovered = jax.nn.relu(jnp.matmul(x_compressed, tms_weights.matrix) + tms_weights.bias)

    return x_recovered

def loss(w, x):
    x_pred = forward_pass(w, x)
    return jnp.mean((x - x_pred) ** 2)

def save_model(filepath, weights):

    if not filepath.endswith('.npz'):
        filepath = filepath + '.npz'

    jnp.savez(filepath, matrix=weights.matrix, bias=weights.bias)

def load_model(filepath):
    data = jnp.load(filepath)
    weights = TMSWeights(
        matrix=data['matrix'], 
        bias=data['bias']
    )
    return weights
