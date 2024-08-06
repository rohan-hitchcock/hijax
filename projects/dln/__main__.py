import argparse
from typing import Literal

import jax
import jax.numpy as jnp

from dln.model import DeepLinearNetwork
from dln import llc
from dln.train import generate_dataset, prepare_data, generate_data
from shared.samplers_optax import SGLD

import matplotlib.pyplot as plt


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
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--num_chains", type=int, default=1)
    parser.add_argument("--burnin_prop", type=float, default=0.9)


    args = parser.parse_args()

    key = jax.random.key(args.seed)

    num_layers_min, num_layers_max, min_width, max_width, epsilon, num_steps, n = get_params(args.oom)

    if args.epsilon is not None:
        epsilon = args.epsilon

    nbeta = llc.nbeta(args.beta, n)
    
    sampler = SGLD(epsilon, args.gamma, nbeta, args.seed)

    sample_llc_multichain = jax.jit(jax.vmap(llc.sample_llc, in_axes=(None, None, 0)), static_argnames=['sampler'])
    
    loss_fn = lambda m, x, y : jnp.mean((m(x) - y) ** 2)
    val_grad_loss_fn = jax.value_and_grad(loss_fn)

    for eid in tqdm.trange(args.num_experiments):
        
        key, key_model = jax.random.split(key)
        init_model = DeepLinearNetwork.initialize_true(key_model, num_layers_min, num_layers_max, min_width, max_width)

        # print(f"{init_model.dim_in=}")
        # print(f"{init_model.dim_out=}")


        key, key_data = jax.random.split(key)
        data = generate_data(key_data, n, init_model.dim_in)


        # print(f"{data.shape=}")
        outputs = init_model(data)

        # print(f"{outputs.shape=}")

        


        key, key_dataset = jax.random.split(key)
        inputs, targets = prepare_data(key_dataset, data, outputs, num_steps, args.batch_size)
        # print(f"{dataset.shape=}")


        l = jax.vmap(loss_fn, in_axes=(None, 0, 0))(init_model, inputs, targets).mean()
        # print("Mean loss ", l)

    
        # loss_fn = lambda updated_model, x : ((init_model(x) - updated_model(x)) ** 2).mean()
        

        key, key_sampler = jax.random.split(key)

        optimizer_state = sampler.init(init_model, key_sampler)
        model = init_model
        trace = []
        for i, (x, y) in enumerate(zip(inputs, targets)):

            loss, grad_loss = val_grad_loss_fn(model, x, y)
            updates, optimizer_state = sampler.update(grad_loss, optimizer_state, model)
            model = jax.tree.map(lambda p, u : p + u, model, updates)
            
            trace.append(loss)
            if jnp.isnan(loss):
                break
            # model = jax.tree.map(lambda w, g: w - 0.00435 * g, model, grad_loss)
            

            """print(f"min_grad: {min_tree(grad_loss)}, max_grad={max_tree(grad_loss)}")
            print(f"min_update: {min_tree(updates)}, max_update={max_tree(updates)}")
            print(f"{loss.item()=}")

            if jnp.isnan(loss) or i > 30:
                exit()"""


        """
        traces = sample_llc_multichain(model, sampler, data)
        estimated_llc_per_chain = jax.vmap(llc.estimate_llc, in_axes=(0, None, None))(traces, int(args.burnin_prop * num_steps), nbeta)

        nan_mask = ~jnp.isnan(estimated_llc_per_chain)
        estimated_llc = jnp.mean(estimated_llc_per_chain, where=nan_mask)


        
        traces = llc.sample_llc(model, sampler, data[0])
        
        """

        trace = jnp.array(trace)
        estimated_llc = llc.estimate_llc(trace, int(args.burnin_prop * num_steps), nbeta)

        moving_llc_estimate = llc.llc_moving_mean(trace[0], trace, nbeta)
        fig, ax = plt.subplots()
        ax.plot(trace)
        plt.savefig(f"debug/exp={eid}.png")
        plt.close(fig)

        theoretical_llc = init_model.theoretical_llc()


        print(f"Exp {eid}: ", model.num_parameters, init_model.rank(), theoretical_llc, estimated_llc)
        

        


    
