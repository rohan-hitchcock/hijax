import jax 
import jax.numpy as jnp
import optax
from optax import GradientTransformation
from jaxtyping import Array, Key
from functools import partial


def sample_llc(key: Key, model, sampler, data: Array):

    init_model = model

    loss_fn = lambda updated_model, x : jnp.mean(((init_model(x) - updated_model(x)) ** 2))
    val_grad_loss_fn = jax.value_and_grad(loss_fn)
    
    def step(carry, x):

        model, optimizer_state = carry
        loss, grad_loss = val_grad_loss_fn(model, x)
        updates, optimizer_state = sampler.update(grad_loss, optimizer_state, model)
        model = jax.tree.map(lambda p, u : p + u, model, updates)

        return (model, optimizer_state), loss


    opt_state = sampler.init(model, key)
    _, trace = jax.lax.scan(step, (init_model, opt_state), data)

    return trace

def llc_mean(init_loss, loss_trace, nbeta):
    return (jnp.mean(loss_trace) - init_loss) * nbeta 


def llc_moving_mean(init_loss, loss_trace, beta):
    moving_mean_loss = jnp.cumsum(loss_trace) / (jnp.arange(len(loss_trace)) + 1)
    return (moving_mean_loss - init_loss) * beta


def nbeta(beta, n):
    return beta * (n / jnp.log(n))

def estimate_llc(trace, num_burnin_steps, nbeta):
    return llc_mean(trace[0], trace[num_burnin_steps:], nbeta)

