import os

import jax
import tqdm

import tyro
from functools import partial


from shared.utils import ExponentialMovingAverage
from tms.model import TMSModel, loss_fn
from shared.samplers import gradient_descent_step
from tms.data import generate_dataset


@partial(jax.jit, static_argnames=['loss_fn'])
def train_step(model, loss_fn, batch, learning_rate):
    l, g = jax.value_and_grad(loss_fn)(model, batch)
    model = gradient_descent_step(model, g, learning_rate)
    return model, l

def train(key, num_steps: int, batch_size: int = 32, in_dim: int = 5, hidden_dim: int = 2, learning_rate: float = 0.01, checkpoint_rate: int = 100, checkpoint_dir: str = None):

    key_data, key = jax.random.split(key)

    data = generate_dataset(key_data, in_dim, batch_size, num_steps)

    key_weight, key = jax.random.split(key)
    model = TMSModel.initialize(key_weight, in_dim, hidden_dim)
    # model = TMSModel.initialize_triangle(key_weight, in_dim)


    ema_loss = ExponentialMovingAverage()
    with tqdm.tqdm(enumerate(data), total=len(data), desc='Training', unit='step') as pbar:
        for step, batch in pbar:

            model, l = train_step(model, loss_fn, batch, learning_rate)

            ema_loss.update(l)
            if checkpoint_dir is not None and step % checkpoint_rate == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'step={step}.npz')
                model.save(checkpoint_path)
            
            pbar.set_postfix({'loss': f'{ema_loss:.3f}'})

    return model



if __name__ == "__main__":
    tyro.cli(train)
