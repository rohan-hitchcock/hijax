import jax
import jax.numpy as jnp

from samplers import sgld_step


def sgld_chain(init_weight, data, loss_fn, learning_rate, gamma, beta):


    val_grad_loss_fn = jax.value_and_grad(loss_fn)

    
    def chain_step(w, x):

        loss, grad_loss = val_grad_loss_fn(w, x)
        w = sgld_step(w, init_weight, grad_loss, learning_rate, gamma, beta)

        return w, loss

    trace = jax.lax.scan(chain_step, init_weight, data)

    return trace


def llc_mean(init_loss, loss_trace, beta):
    return beta * (jnp.mean(loss_trace) - init_loss)


def llc_moving_mean(init_loss, loss_trace, beta):
    moving_mean_loss = jnp.cumsum(loss_trace) / (jnp.arange(len(loss_trace)) + 1)
    return beta * (moving_mean_loss - init_loss)


def sgld_multichain(init_weight, data_by_chain, loss_fn, learning_rate, gamma, beta):

    return jax.vmap(sgld_multichain, in_axes=(None, 0, None, None, None, None))(
        init_weight=init_weight, 
        data=data_by_chain, 
        loss_fn=loss_fn, 
        learning_rate=learning_rate, 
        gamma=gamma, 
        beta=beta,
    )
