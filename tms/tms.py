
import jax
import jax.numpy as jnp
import tqdm

import plotille

def init_weights(key, in_dim, hidden_dim):
    
    bias = jnp.zeros(in_dim)

    bound = jnp.sqrt(1 / in_dim)
    matrix = jax.random.uniform(key, shape=(hidden_dim, in_dim), minval=-bound, maxval=bound)

    return (matrix, bias)


def forward_pass(w, x):

    matrix, bias = w

    x_compressed = jnp.matmul(x, matrix.T)
    x_recovered = jax.nn.relu(jnp.matmul(x_compressed, matrix) + bias)

    return x_recovered


def loss(w, x):
    x_pred = forward_pass(w, x)
    return jnp.mean((x - x_pred) ** 2)


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

def train(seed, num_steps, batch_size, in_dim, hidden_dim, learning_rate, logging_rate=100):

    key = jax.random.key(seed)
    key_data, key = jax.random.split(key)

    data = generate_dataset(key_data, in_dim, batch_size, num_steps)

    key_weight, key = jax.random.split(key)
    weights = init_weights(key_weight, in_dim, hidden_dim)

    val_grad_loss = jax.value_and_grad(loss)


    ema_loss = ExponentialMovingAverage(decay=0.95)
    with tqdm.tqdm(enumerate(data), total=len(data)) as pbar:
        for step, batch in pbar:
            l, g = val_grad_loss(weights, batch)
            ema_loss.update(l)

            grad_matrix, grad_bias = g
            matrix, bias = weights
            weights = (matrix - learning_rate * grad_matrix, bias - learning_rate * grad_bias)

            if step % logging_rate == 0:
                pbar.write(vis_weights(weights))        
            
            pbar.set_postfix({'loss': f'{ema_loss:.3f}'})

    return weights



def vis_weights(weights, overwrite=True):

    matrix, biases = weights

    fig = plotille.Figure()
    fig.width = 40
    fig.height = 15
    fig.set_x_limits(-1, 1)
    fig.set_y_limits(-1, 1)

    for col in matrix.T:
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

    seed = 1000
    batch_size = 8
    in_dim = 5
    hidden_dim = 2
    learning_rate = 0.01
    num_steps = 10000
    batch_size = 32

    train(seed, num_steps, batch_size, in_dim, hidden_dim, learning_rate)







