import os

import jax
import tqdm

import tyro
from functools import partial


from tms.model import TMSModel, loss_fn
from tms.samplers import gradient_descent_step
from tms.data import generate_dataset


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
