from typing import Tuple

import jax
from jaxtyping import PRNGKeyArray as Key, Array, Float

from tqdm import tqdm

from dln.model import DeepLinearNetwork
from shared.utils import ExponentialMovingAverage
from shared.samplers import gradient_descent_step

def train_step(
        model: DeepLinearNetwork, 
        true_model: DeepLinearNetwork, 
        batch: Float[Array, 'batch_size in_dim'], 
        learning_rate: float
        ) -> Tuple[DeepLinearNetwork, float]:

    y = true_model(batch)
    loss, grad_loss = jax.value_and_grad(model.loss)(batch, y)
    model = gradient_descent_step(model, grad_loss, learning_rate)
    return model, loss


def train(model: DeepLinearNetwork, true_model: DeepLinearNetwork, dataset: Float[Array, 'num_elem batch_size in_dim'], learning_rate: float):

    ema_loss = ExponentialMovingAverage()
    with tqdm(enumerate(dataset), total=len(dataset), desc='Training', unit='step') as pbar:
        for step, batch in pbar:

            model, l = train_step(model, true_model, batch, learning_rate)

            ema_loss.update(l)

            """if checkpoint_dir is not None and step % checkpoint_rate == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'step={step}.npz')
                model.save(checkpoint_path)"""
            
            pbar.set_postfix({'loss': f'{ema_loss:.3f}'})

    return model


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

