from typing import Tuple

import jax
from jaxtyping import PRNGKeyArray as Key, Array, Float

from tqdm import tqdm

from dln.model import DeepLinearNetwork
from shared.utils import ExponentialMovingAverage, TrainLogger

import optax
from optax import GradientTransformation

from functools import partial

def train(model: DeepLinearNetwork, true_model: DeepLinearNetwork, dataset: Float[Array, 'num_elem batch_size in_dim'], optimizer: GradientTransformation, checkpoint_rate: int = 100):


    optimizer_state = optimizer.init(model)



    with TrainLogger(None, 100, len(dataset), enable_plot=False) as logger:
        for step, x in enumerate(dataset):

            model, loss = train_step(model, true_model, optimizer, optimizer_state, x)

            logger.log_step(step, loss, model)

    return model

@partial(jax.jit, static_argnames=['optimizer'])
def train_step(model, true_model, optimizer, optimizer_state, x):
    y_true = true_model(x)
    loss, grad_loss = jax.value_and_grad(model.loss)(model, x, y_true)
    updates, optimizer_state = optimizer.update(grad_loss, optimizer_state, model)
    model = optax.apply_updates(model, updates)
    return model, loss


def generate_dataset(
        key: Key, 
        num_elements: int, 
        batch_size: int, 
        in_dim: int, 
        min_val: float = -10.0, 
        max_val: float = 10.0,
    ) -> Float[Array, '{num_elements} {batch_size} {in_dim}']:
    
    return jax.random.uniform(
        key, 
        shape=(num_elements, batch_size, in_dim), 
        minval=min_val, 
        maxval=max_val
    )


def generate_data(key: Key, num_elements: int, dim: int, min_val: float = -10.0, max_val: float = 10.0):
    return jax.random.uniform(key, shape=(num_elements, dim), minval=min_val, maxval=max_val)


def prepare_data(key: Key, data: Float[Array, 'num_elem dim_in'], outputs, num_steps: int, batch_size: int) -> Float[Array, 'num_batches batch_size dim']:

    num_elem, dim_in = data.shape
    steps_per_epoch = num_elem // batch_size
    num_epochs = num_steps // steps_per_epoch

    


    def make_epoch(k, xs, ys):

        shuffle_indices = jax.random.permutation(k, xs.shape[0])

        xs_shuffled = xs[shuffle_indices]
        ys_shuffled = ys[shuffle_indices]

        dim_xs = xs.shape[-1]
        dim_ys = ys.shape[-1]

        return xs_shuffled.reshape(steps_per_epoch, batch_size, dim_xs), ys_shuffled.reshape(steps_per_epoch, batch_size, dim_ys)


    keys_shuffle = jax.random.split(key, num=num_epochs)
    epochs_xs, epochs_ys = jax.vmap(make_epoch, in_axes=(0, None, None))(keys_shuffle, data, outputs)
    


    dim_out = epochs_ys.shape[-1]
    return epochs_xs.reshape(-1, batch_size, dim_in), epochs_ys.reshape(-1, batch_size, dim_out)

    







if __name__ == "__main__":

    seed = 0

    key = jax.random.key(seed)
    key_init_true, key_init_train, key = jax.random.split(key, num=3)


    true_model = DeepLinearNetwork.initialize_true(key_init_true, min_layers=5, max_layers=10, min_layer_size=10, max_layer_size=20)
    model = DeepLinearNetwork.initialize(key_init_train, true_model.layer_sizes)


    optimizer = optax.sgd(learning_rate=0.0001)

    num_steps = 10000
    batch_size = 512
    key_dataset, key = jax.random.split(key)
    dataset = generate_dataset(key_dataset, num_steps, batch_size, in_dim=true_model.layer_sizes[0])

    print(true_model.num_parameters)
    print(dataset.shape)

    
    train(model, true_model, dataset, optimizer)

