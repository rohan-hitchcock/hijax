import jax
import jax.numpy as jnp

from tms.samplers import sgld_step
from functools import partial

def sgld_chain(key, init_weight, data, loss_fn, learning_rate, gamma, beta):


    val_grad_loss_fn = jax.value_and_grad(loss_fn)

    
    def chain_step(carry, x):

        w, key = carry
        key, key_step = jax.random.split(key)

        loss, grad_loss = val_grad_loss_fn(w, x)

        w = sgld_step(key_step, w, init_weight, grad_loss, learning_rate, gamma, beta)

        return (w, key), loss

    _, trace = jax.lax.scan(chain_step, (init_weight, key), data)

    return trace


def llc_mean(init_loss, loss_trace, beta):
    return beta * (jnp.mean(loss_trace) - init_loss)


def llc_moving_mean(init_loss, loss_trace, beta):
    moving_mean_loss = jnp.cumsum(loss_trace) / (jnp.arange(len(loss_trace)) + 1)
    return beta * (moving_mean_loss - init_loss)

@partial(jax.jit, static_argnames=['loss_fn', 'num_burnin_steps'])
def estimate_llc(key, init_weight, data_by_chain, loss_fn, epsilon, gamma, beta, num_burnin_steps):

    # compute multiple different chains    
    keys = jax.random.split(key, num=data_by_chain.shape[0])
    loss_traces = jax.vmap(sgld_chain, (0, None, 0, None, None, None, None))(
        keys,
        init_weight, 
        data_by_chain, 
        loss_fn, 
        epsilon, 
        gamma, 
        beta,
    )

    init_losses = loss_traces[:, 0]
    draws = loss_traces[:, num_burnin_steps:]
    llcs = jax.vmap(llc_mean, in_axes=(0, 0, None))(init_losses, draws, beta)

    return llcs, loss_traces
