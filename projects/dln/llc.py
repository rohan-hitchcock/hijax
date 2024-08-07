import jax 
import jax.numpy as jnp

from jaxtyping import Array, Key, Float

from functools import partial


def sample_llc(key: Key, model, sampler, loss_fn, xs: Array, ys: Array):

    init_model = model

    
    val_grad_loss_fn = jax.value_and_grad(loss_fn)
    
    def step(carry, elem):

        x, y = elem
        model, optimizer_state = carry

        loss, grad_loss = val_grad_loss_fn(model, x, y)
        updates, optimizer_state = sampler.update(grad_loss, optimizer_state, model)
        model = jax.tree.map(lambda p, u : p + u, model, updates)

        return (model, optimizer_state), loss


    opt_state = sampler.init(model, key)
    _, trace = jax.lax.scan(step, (init_model, opt_state), (xs, ys))

    return trace


def sample_weights(key: Key, model, sampler, loss_fn, xs: Array, ys: Array):

    init_model = model

    
    val_grad_loss_fn = jax.value_and_grad(loss_fn)
    
    def step(carry, elem):

        x, y = elem
        model, optimizer_state = carry

        loss, grad_loss = val_grad_loss_fn(model, x, y)
        updates, optimizer_state = sampler.update(grad_loss, optimizer_state, model)
        model = jax.tree.map(lambda p, u : p + u, model, updates)

        return (model, optimizer_state), (loss, model)


    opt_state = sampler.init(model, key)
    _, (loss_trace, weight_trace) = jax.lax.scan(step, (init_model, opt_state), (xs, ys))

    return loss_trace, weight_trace

@partial(jax.jit, static_argnames=['sampler', 'loss_fn', 'num_chains', 'num_steps', 'batch_size'])
def sample_llc_multichain(key: Key, model, sampler, loss_fn, xs: Array, ys: Array, num_chains, num_steps, batch_size):

    prepare_data_multichain = jax.vmap(prepare_data, in_axes=(0, None, None, None, None))

    sample_multichain = jax.vmap(sample_llc, in_axes=(0, None, None, None, 0, 0))

    keys = jax.random.split(key, num=num_chains + 1)
    key, keys_chain_datasets = keys[0], keys[1:]
    batched_xs_by_chain, batched_ys_by_chain = prepare_data_multichain(keys_chain_datasets, xs, ys, num_steps, batch_size)

    keys_chains = jax.random.split(key, num=num_chains + 1)
    key, keys_chains = keys[0], keys[1:]
        
    traces = sample_multichain(keys_chains, model, sampler, loss_fn, batched_xs_by_chain, batched_ys_by_chain)

    return traces

# @partial(jax.jit, static_argnames=['sampler', 'loss_fn', 'num_chains', 'num_steps', 'batch_size'])
def sample_weights_multichain(key: Key, model, sampler, loss_fn, xs: Array, ys: Array, num_chains, num_steps, batch_size):

    prepare_data_multichain = jax.vmap(prepare_data, in_axes=(0, None, None, None, None))

    sample_multichain = jax.vmap(sample_weights, in_axes=(0, None, None, None, 0, 0))

    keys = jax.random.split(key, num=num_chains + 1)
    key, keys_chain_datasets = keys[0], keys[1:]
    batched_xs_by_chain, batched_ys_by_chain = prepare_data_multichain(keys_chain_datasets, xs, ys, num_steps, batch_size)

    keys_chains = jax.random.split(key, num=num_chains + 1)
    key, keys_chains = keys[0], keys[1:]
        
    traces = sample_multichain(keys_chains, model, sampler, loss_fn, batched_xs_by_chain, batched_ys_by_chain)

    return traces



def prepare_data(key: Key, xs, ys, num_steps: int, batch_size: int):

    num_elem, dim_in = xs.shape
    steps_per_epoch = num_elem // batch_size
    num_epochs = num_steps // steps_per_epoch

    def make_epoch(k, xs, ys):

        shuffle_indices = jax.random.permutation(k, xs.shape[0])

        xs_shuffled = xs[shuffle_indices]
        ys_shuffled = ys[shuffle_indices]

        dim_xs = xs.shape[-1]
        dim_ys = ys.shape[-1]

        return xs_shuffled.reshape(steps_per_epoch, batch_size, dim_xs), ys_shuffled.reshape(steps_per_epoch, batch_size, dim_ys)


    keys_shuffle = jax.random.split(key, num=num_epochs)
    epochs_xs, epochs_ys = jax.vmap(make_epoch, in_axes=(0, None, None))(keys_shuffle, xs, ys)
    
    dim_out = epochs_ys.shape[-1]
    return epochs_xs.reshape(-1, batch_size, dim_in), epochs_ys.reshape(-1, batch_size, dim_out)

def llc_mean(init_loss, loss_trace, nbeta):
    return (jnp.mean(loss_trace) - init_loss) * nbeta 


def llc_moving_mean(init_loss, loss_trace, beta):
    moving_mean_loss = jnp.cumsum(loss_trace) / (jnp.arange(len(loss_trace)) + 1)
    return (moving_mean_loss - init_loss) * beta


def nbeta(beta, n):
    return beta * (n / jnp.log(n))

def estimate_llc(trace, num_burnin_steps, nbeta):
    return llc_mean(trace[0], trace[num_burnin_steps:], nbeta)

