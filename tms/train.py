import jax
import jax.numpy as jnp
import tqdm

import tyro
import plotille

import model
from samplers import gradient_descent_step
from data import generate_dataset

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

def train(seed: int, num_steps: int, batch_size: int = 32, in_dim: int = 5, hidden_dim: int = 2, learning_rate: float = 0.01, logging_rate: int = 100):

    key = jax.random.key(seed)
    key_data, key = jax.random.split(key)

    data = generate_dataset(key_data, in_dim, batch_size, num_steps)

    key_weight, key = jax.random.split(key)
    weights = model.init_weights(key_weight, in_dim, hidden_dim)

    val_grad_loss = jax.value_and_grad(model.loss)

    ema_loss = ExponentialMovingAverage()
    print(vis_weights(weights, overwrite=False))
    with tqdm.tqdm(enumerate(data), total=len(data)) as pbar:
        for step, batch in pbar:
            l, g = val_grad_loss(weights, batch)
            ema_loss.update(l)

            weights = gradient_descent_step(weights, g, learning_rate)


            if step % logging_rate == 0:
                pbar.write(vis_weights(weights))        
            
            pbar.set_postfix({'loss': f'{ema_loss:.3f}'})

    return weights

def vis_weights(weights, overwrite=True):

    fig = plotille.Figure()
    fig.width = 40
    fig.height = 15
    fig.set_x_limits(-1, 1)
    fig.set_y_limits(-1, 1)

    for col in weights.matrix.T:
        xs, ys = line(col, samples=2)
        fig.plot(xs, ys)

    figure_str = str(fig.show())
    reset = f"\x1b[{len(figure_str.splitlines())}A" if overwrite else ""
    return reset + figure_str


def line(end, start=None, samples=50):

    if start is None:
        start = jnp.zeros_like(end)


    start = start[:,jnp.newaxis]
    end = end[:,jnp.newaxis]

    alpha = jnp.linspace(0, 1, num=samples)[jnp.newaxis,:]

    return alpha * start + (1 - alpha) * end

if __name__ == "__main__":
    tyro.cli(train)
