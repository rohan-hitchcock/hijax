"""
Teacher-student perceptron learning, vanilla JAX.
"""

import time

import plotille
import tqdm
import tyro

import jax
import jax.numpy as jnp


# # # 
# Training loop


def main(
    num_steps: int = 200,
    learning_rate: float = 0.01,
    seed: int = 0,
):
    key = jax.random.key(seed)

    key_init_student, key = jax.random.split(key)
    w = init_params(key_init_student)

    key_init_teacher, key = jax.random.split(key)
    w_star = init_params(key_init_teacher)

    print(vis(student=w, teacher=w_star, overwrite=False))

    val_grad_loss = jax.value_and_grad(loss)

    for t in tqdm.trange(num_steps):
        
        # sample a singe data point (i.e. original SGD, not minibatch)
        key_data_t, key = jax.random.split(key)
        x = jax.random.normal(key_data_t)

        # compute the gradient of the loss wrt w
        l, g = val_grad_loss(w, w_star, x)
        
        # update w 
        w = (w[0] - learning_rate * g[0], w[1] - learning_rate * g[1])


        # update our visualisation
        tqdm.tqdm.write(vis(student=w, teacher=w_star, x=x))
        tqdm.tqdm.write(
            f'x: {x:+.3f} | loss: {l:.3f} | '
            + f'a {w[0]:+.3f} | b {w[1]:+.3f} '
            + f'a* {w_star[0]:+.3f} | b* {w_star[1]:+.3f} '
        )

        # time.sleep(0.5)
        

def loss(w, w_star, x):
    y = forward_pass(w, x)
    y_star = forward_pass(w_star, x)

    # mean is not needed here (has no effect, but would be used in minibatch)
    loss = jnp.mean((y - y_star) ** 2)
    return loss


def init_params(key):
    """ Idiomatic in deep learning with jax. Used to initialise the parameters 
        of a network
    """

    key_a, key = jax.random.split(key)
    a = jax.random.normal(key_a)

    key_b, key = jax.random.split(key)
    b = jax.random.normal(key_b)

    # Or...
    # a, b = jax.random.normal(key, shape=(2,))

    return a, b

def forward_pass(w, x):
    """ Idiomatic in deep learning with jax. Used to define how the model works 
        on a single input `x`. Here `w` is the weights of the network.
    """
    a, b = w        # in this problem we have two weights
    return a * x + b

# # # 
# Perceptron architecture


# TODO!


# # # 
# Visualisation


def vis(x=None, overwrite=True, **models):
    # configure plot
    fig = plotille.Figure()
    fig.width = 40
    fig.height = 15
    fig.set_x_limits(-4, 4)
    fig.set_y_limits(-3, 3)
    
    # compute data and add to plot
    xs = jnp.linspace(-4, 4)
    for (label, w), color in zip(models.items(), ['cyan', 'magenta']):
        ys = forward_pass(w, xs)
        fig.plot(xs, ys, label=label, lc=color)
    
    # add a marker for the input batch
    if x is not None:
        fig.text([x], [0], ['x'], lc='yellow')
    
    # render to string
    figure_str = str(fig.show(legend=True))
    reset = f"\x1b[{len(figure_str.splitlines())+1}A" if overwrite else ""
    return reset + figure_str


# # # 
# Entry point


if __name__ == "__main__":
    tyro.cli(main)
