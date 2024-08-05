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

