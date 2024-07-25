import jax.numpy as jnp
import plotille


def line(end, start=None, samples=50):

    if start is None:
        start = jnp.zeros_like(end)

    start = start[:,jnp.newaxis]
    end = end[:,jnp.newaxis]

    alpha = jnp.linspace(0, 1, num=samples)[jnp.newaxis,:]

    return alpha * start + (1 - alpha) * end


def get_weights_figure(weights):

    fig = plotille.Figure()
    fig.width = 40
    fig.height = 15
    fig.set_x_limits(-1, 1)
    fig.set_y_limits(-1, 1)

    for col in weights.matrix.T:
        xs, ys = line(col, samples=2)
        fig.plot(xs, ys)

    figure_str = str(fig.show())
    return figure_str

def add_overwrite(figure_str):
    return f"\x1b[{len(figure_str.splitlines())}A" + figure_str
