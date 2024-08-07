import argparse
from typing import Literal
import os
import json
import jax
import jax.numpy as jnp
import math
from dln.model import DeepLinearNetwork
from dln import llc
from dln.train import prepare_data, generate_data
from shared.samplers_optax import SGLD
from shared.utils import create_unique_subdirectory

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
import optax

import tqdm


OUTPUT_DIR = "./dln/results"

NUM_SAVED_TRACES = 9

def get_params(oom_str: Literal['1K', '10K', '100K', '1M', '10M', '100M']):
    """ Parameters from Furman and Lau. See Appendix E.2"""
    if oom_str == '1K':
        num_layers_min = 2
        num_layers_max = 5
        min_width = 5
        max_width = 50
        epsilon = 5e-7
        num_steps = 10000
        dataset_size = 10 ** 5

    elif oom_str == '10K':
        num_layers_min = 2
        num_layers_max = 10
        min_width = 5
        max_width = 100
        epsilon = 5e-7
        num_steps = 10000
        dataset_size = 10 ** 5

    elif oom_str == '100K':
        num_layers_min = 2
        num_layers_max = 10
        min_width = 50
        max_width = 500
        epsilon = 1e-7
        num_steps = 50000
        dataset_size = 10 ** 6

    elif oom_str == '1M':
        num_layers_min = 2
        num_layers_max = 20
        min_width = 100
        max_width = 1000
        epsilon = 5e-8
        num_steps = 50000
        dataset_size = 10 ** 6

    elif oom_str == '10M':
        num_layers_min = 2
        num_layers_max = 20
        min_width = 500
        max_width = 2000
        epsilon = 2e-8
        num_steps = 50000
        dataset_size = 10 ** 6

    elif oom_str == '100M':
        num_layers_min = 2
        num_layers_max = 40
        min_width = 500
        max_width = 3000
        epsilon = 2e-8
        num_steps = 50000
        dataset_size = 10 ** 6

    else:
        raise ValueError(f"'{oom_str}' not recognised")

    return num_layers_min, num_layers_max, min_width, max_width, epsilon, num_steps, dataset_size


def plot_saved_traces(saved_traces, figpath=None):
    grid_dim = int(math.sqrt(NUM_SAVED_TRACES))
    fig, axes = plt.subplots(nrows=grid_dim, ncols=grid_dim)

    axes = axes.flatten()

    for ax, traces in zip(axes, saved_traces):
        for trace in traces:
            ax.plot(trace)

    if figpath is not None:
        plt.savefig(figpath)
    else:
        plt.show()

    plt.close(fig)


def plot_results(results_df, figpath=None):

    fig, ax = plt.subplots()

    ax.scatter(results_df['true_llc'], results_df['estimated_llc'])

    
    ax_lower_lim = 0
    ax_upper_lim = max(results_df['true_llc'].max(), results_df['estimated_llc'].max())
    
    line = np.linspace(ax_lower_lim, ax_upper_lim)
    ax.plot(line, line, color='k', alpha=0.5)

    ax.set_xlabel('True LLC')
    ax.set_ylabel('Estimated LLC')

    ax.set_xlim(left=ax_lower_lim, right=ax_upper_lim)
    ax.set_ylim(bottom=ax_lower_lim, top=ax_upper_lim)

    if figpath is not None:
        plt.savefig(figpath)
    else:
        plt.show()

    plt.close(fig)

    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--oom", required=True, choices=['1K', '10K', '100K', '1M', '10M', '100M'], help="Order of magnitude of models")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_experiments", type=int, default=100)
    
    parser.add_argument("--epsilon", type=float, default=None, help="If set, overwrites default epsilon for given oom")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--num_chains", type=int, default=1)
    parser.add_argument("--burnin_prop", type=float, default=0.9, help="Proportion of SGLD steps to use for burn-in")

    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)


    args = parser.parse_args()

    key = jax.random.key(args.seed)

    num_layers_min, num_layers_max, min_width, max_width, epsilon, num_steps, dataset_size = get_params(args.oom)

    num_burnin_steps = int(args.burnin_prop * num_steps)

    epsilon = args.epsilon if args.epsilon is not None else epsilon

    nbeta = llc.nbeta(args.beta, dataset_size)
    sampler = SGLD(epsilon, args.gamma, nbeta)
    
    loss_fn = lambda m, x, y : jnp.mean((m(x) - y) ** 2)
    
    
    trace_save_freq = args.num_experiments // NUM_SAVED_TRACES
    saved_traces = []

    results = []
    for i in tqdm.trange(args.num_experiments):
        
        key, key_model = jax.random.split(key)
        model = DeepLinearNetwork.initialize_true(key_model, num_layers_min, num_layers_max, min_width, max_width)
        
        rank = model.rank()

        key, key_data = jax.random.split(key)
        xs = generate_data(key_data, dataset_size, model.dim_in)
        ys = model(xs)

        key, key_sampling = jax.random.split(key)
        traces = llc.sample_llc_multichain(key_sampling, model, sampler, loss_fn, xs, ys, args.num_chains, num_steps, args.batch_size)

        
        # L_n(w^*) = 0 here, so \lambda = n\beta E[L_n(w)]
        estimated_llc_per_chain = jax.vmap(lambda trace : nbeta * jnp.mean(trace))(traces[:,num_burnin_steps:])

        
        results.append({
            'num_params': model.num_parameters, 
            'true_llc': model.learning_coefficient(), 
            'rank': model.rank().item(), 
            'estimated_llc': jnp.mean(estimated_llc_per_chain, where=~jnp.isnan(estimated_llc_per_chain)),
        } | {
            f'llc_chain_{i}': llc.item() 
            for i, llc in enumerate(estimated_llc_per_chain)
        }
        )

        if i % trace_save_freq == 0 and len(saved_traces) < NUM_SAVED_TRACES:
            saved_traces.append(traces)



    os.makedirs(args.output_dir, exist_ok=True)

    exp_dir = create_unique_subdirectory(args.output_dir)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(exp_dir, 'results.csv'), index=False)

    plot_results(results_df, figpath=os.path.join(exp_dir, 'results.png'))

    jnp.savez(os.path.join(exp_dir, 'sample_traces.npz'), *saved_traces)

    with open(os.path.join(exp_dir, 'args.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4)

    plot_saved_traces(saved_traces, figpath=os.path.join(exp_dir, 'sample_traces.png'))
