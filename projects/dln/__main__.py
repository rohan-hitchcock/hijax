import argparse
from typing import Literal

import jax
import jax.numpy as jnp

from dln.model import DeepLinearNetwork
from dln import llc
from dln.train import generate_dataset
from shared.samplers_optax import SGLD

import optax

import tqdm

def get_params(oom_str: Literal['1K', '10K', '100K', '1M', '10M', '100M']):

    if oom_str == '1K':
        num_layers_min = 2
        num_layers_max = 5
        min_width = 5
        max_width = 50
        epsilon = 5e-7
        num_steps = 10000
        n = 10 ** 5

    elif oom_str == '10K':
        num_layers_min = 2
        num_layers_max = 10
        min_width = 5
        max_width = 100
        epsilon = 5e-7
        num_steps = 10000
        n = 10 ** 5

    elif oom_str == '100K':
        num_layers_min = 2
        num_layers_max = 10
        min_width = 50
        max_width = 500
        epsilon = 1e-7
        num_steps = 50000
        n = 10 ** 6

    elif oom_str == '1M':
        num_layers_min = 2
        num_layers_max = 20
        min_width = 100
        max_width = 1000
        epsilon = 5e-8
        num_steps = 50000
        n = 10 ** 6

    elif oom_str == '10M':
        num_layers_min = 2
        num_layers_max = 20
        min_width = 500
        max_width = 2000
        epsilon = 2e-8
        num_steps = 50000
        n = 10 ** 6

    elif oom_str == '100M':
        num_layers_min = 2
        num_layers_max = 40
        min_width = 500
        max_width = 3000
        epsilon = 2e-8
        num_steps = 50000
        n = 10 ** 6

    else:
        raise ValueError(f"'{oom_str}' not recognised")

    return num_layers_min, num_layers_max, min_width, max_width, epsilon, num_steps, n



def min_tree(pytree):
    flat_tree, _ = jax.tree.flatten(pytree)
    return min(leaf.min() for leaf in flat_tree).item()

def max_tree(pytree):
    flat_tree, _ = jax.tree.flatten(pytree)
    return max(leaf.max() for leaf in flat_tree).item()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--oom", required=True, choices=['1K', '10K', '100K', '1M', '10M', '100M'])
    parser.add_argument("--num_experiments", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_chains", type=int, default=1)
    parser.add_argument("--burnin_prop", type=float, default=0.9)


    args = parser.parse_args()

    key = jax.random.key(args.seed)

    num_layers_min, num_layers_max, min_width, max_width, epsilon, num_steps, n = get_params(args.oom)
    nbeta = llc.nbeta(args.beta, n)
    # nbeta = llc.nbeta(args.beta, args.batch_size)
    sampler = SGLD(epsilon, args.gamma, nbeta, args.seed)

    sample_llc_multichain = jax.jit(jax.vmap(llc.sample_llc, in_axes=(None, None, 0)), static_argnames=['sampler'])
    # sample_llc_multichain = jax.vmap(llc.sample_llc, in_axes=(None, None, 0))
    

    

    for _ in tqdm.trange(args.num_experiments):
        
        key, key_model = jax.random.split(key)
        init_model = DeepLinearNetwork.initialize_true(key_model, num_layers_min, num_layers_max, min_width, max_width)

        key, key_data = jax.random.split(key)
        data = generate_dataset(key_data, args.num_chains * num_steps, args.batch_size, init_model.layer_sizes[0])
        data = data.reshape((args.num_chains, num_steps, args.batch_size, -1))


        print(data.shape)
        print(init_model(data[0][0][0]))


        loss_fn = lambda updated_model, x : ((init_model(x) - updated_model(x)) ** 2).mean()
        val_grad_loss_fn = jax.value_and_grad(loss_fn)

        optimizer_state = sampler.init(init_model)
        model = init_model
        for i, x in enumerate(data[0]):

            loss, grad_loss = val_grad_loss_fn(model, x)
            updates, optimizer_state = sampler.update(grad_loss, optimizer_state, model)
            model = optax.apply_updates(model, updates)
            
            # model = jax.tree.map(lambda w, g: w - 0.00435 * g, model, grad_loss)
            

            print(f"min_grad: {min_tree(grad_loss)}, max_grad={max_tree(grad_loss)}")
            print(f"min_update: {min_tree(updates)}, max_update={max_tree(updates)}")
            print(f"{loss.item()=}")

            if jnp.isnan(loss) or i > 30:
                exit()


        
        model = init_model

        traces = sample_llc_multichain(model, sampler, data)
        estimated_llc_per_chain = jax.vmap(llc.estimate_llc, in_axes=(0, None, None))(traces, int(args.burnin_prop * num_steps), nbeta)

        nan_mask = ~jnp.isnan(estimated_llc_per_chain)
        estimated_llc = jnp.mean(estimated_llc_per_chain, where=nan_mask)


        traces = llc.sample_llc(model, sampler, data[0])
        estimated_llc = llc.estimate_llc(traces, int(args.burnin_prop * num_steps), nbeta)

        theoretical_llc = model.theoretical_llc()


        print(model.num_parameters, model.rank(), theoretical_llc, estimated_llc, jnp.count_nonzero(nan_mask))
        print(traces[:30])

        


    
