import jax
import jax.numpy as jnp

from shared.samplers import sgld_step
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


def sgld_multichain(key, init_weight, data_by_chain, loss_fn, epsilon, gamma, beta):
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
    
    return loss_traces


def llc_mean(init_loss, loss_trace, beta):
    return beta * (jnp.mean(loss_trace) - init_loss)


def llc_moving_mean(init_loss, loss_trace, beta):
    moving_mean_loss = jnp.cumsum(loss_trace) / (jnp.arange(len(loss_trace)) + 1)
    return beta * (moving_mean_loss - init_loss)

@partial(jax.jit, static_argnames=['loss_fn', 'num_burnin_steps'])
def estimate_llc(key, init_weight, data_by_chain, loss_fn, epsilon, gamma, beta, num_burnin_steps):

    batch_size = data_by_chain.shape[2]
    beta_corrected = (batch_size / jnp.log(batch_size)) * beta

    # compute multiple different chains    
    loss_traces = sgld_multichain(key, init_weight, data_by_chain, loss_fn, epsilon, gamma, beta_corrected)

    init_losses = loss_traces[:, 0]
    draws = loss_traces[:, num_burnin_steps:]

    llcs = jax.vmap(llc_mean, in_axes=(0, 0, None))(init_losses, draws, beta_corrected)

    return llcs, loss_traces

@partial(jax.jit, static_argnames=['loss_fn'])
def llc_sweep(key, init_weight, data_by_chain, loss_fn, epsilons, gammas, betas):

    batch_size = data_by_chain.shape[2]
    beta_corrected = (batch_size / jnp.log(batch_size)) * betas

    # computes loss traces (one for each chain) for multiple values of beta
    beta_sweep = jax.vmap(sgld_multichain, (0, None, None, None, None, None, 0))
    
    # computes a beta sweep for multiple values of gamma
    gamma_beta_sweep = jax.vmap(beta_sweep, (0, None, None, None, None, 0, None))

    # computes a gamma-beta sweep for multiple values of epsilon
    epsilon_gamma_beta_sweep = jax.vmap(gamma_beta_sweep, (0, None, None, None, 0, None, None))


    sweep_keys = jax.random.split(key, (len(epsilons), len(gammas), len(betas)))

    # result has shape [E, G, B, C, D] where E, G and B are the number of epsilons
    # gammas and betas, and C and D are the number of chains and draws respectively
    return epsilon_gamma_beta_sweep(sweep_keys, init_weight, data_by_chain, loss_fn, epsilons, gammas, beta_corrected)






    

