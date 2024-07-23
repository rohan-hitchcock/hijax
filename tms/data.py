import jax
import jax.numpy as jnp

def generate_vector(key, n):

    key_coord, key = jax.random.split(key)
    key_length, key = jax.random.split(key)

    i = jax.random.randint(key_coord, shape=(), minval=0, maxval=n)
    length = jax.random.uniform(key_length, shape=())

    return jnp.zeros(n).at[i].set(length)

def generate_batch(key, n, batch_size):
    keys = jax.random.split(key, batch_size)
    return jax.vmap(generate_vector, in_axes=(0, None))(keys, n)

def generate_dataset(key, n, batch_size, dataset_size):
    keys = jax.random.split(key, dataset_size)
    return jax.vmap(generate_batch, in_axes=(0, None, None))(keys, n, batch_size)
