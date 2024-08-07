import os
import itertools
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray as Key
import pandas as pd
import json

from shared.utils import register_model

from dln import llc
from shared.samplers_optax import SGLD


import matplotlib.pyplot as plt

from shared.utils import create_unique_subdirectory


OUTPUT_DIR = "dln/results_normal_crossing"

@register_model(weights=['weights'], hparams=['powers'])
class NormalCrossingModel:

    weights: Array
    powers: Array

    def __call__(self, x: Array) -> Array:
        return jnp.prod(jnp.pow(self.weights, self.powers)) * x

    def learning_coefficient(self) -> float:
        # 1/0 := inf in jax
        return jnp.min(0.5 / self.powers)

    @classmethod
    def initialize_true(cls, powers: Array):
        weights = jnp.zeros_like(powers, dtype=jnp.float32)
        return cls(weights=weights, powers=powers)


def run_normal_crossing_experiment(key: Key, powers, sampler, dataset_size, batch_size, num_steps, num_chains, burn_in):
    

    key, key_xs = jax.random.split(key)
    xs = jax.random.normal(key_xs, shape=(dataset_size, 1))

    key, key_ys = jax.random.split(key)
    ys = 0.5 * jax.random.normal(key_ys, shape=(dataset_size, 1)) # N(0, 1/4)

    model_true = NormalCrossingModel.initialize_true(powers)
    
    
    loss_fn = lambda m, x, y : jnp.mean((m(x) - y) ** 2)

    key, key_llc = jax.random.split(key)



    loss_traces, weight_traces = llc.sample_weights_multichain(key_llc, model_true, sampler, loss_fn, xs, ys, num_chains, num_steps, batch_size)



    draws = loss_traces[:,burn_in:]
    estimated_llc = nbeta * (jnp.mean(draws) - loss_fn(model_true, xs, ys))
    

    return {
        'estimated_llc': estimated_llc, 
        'true_llc': model_true.learning_coefficient(), 
        'loss_traces': loss_traces, 
        'weight_traces': weight_traces.weights
    }

    

def plot_loss_traces(loss_traces, title='', figpath=None):
    fig, ax = plt.subplots()
    for trace in loss_traces:
        ax.plot(trace)
    
    if title:
        ax.set_title(title)

    if figpath is None:
        plt.show()   
    else:
        plt.savefig(figpath)

    plt.close(fig)


def plot_true_vs_estimated_llc(results_df, title='', figpath=None):
    fig, ax = plt.subplots()
    
    ax.scatter(results_df['true_llc'], results_df['estimated_llc'])

    ax.set_xlabel('True LLC')
    ax.set_ylabel('Estimated LLC')
    

    min_llc = min(results_df['true_llc'].min(), results_df['estimated_llc'].min(), 0)
    max_llc = max(results_df['true_llc'].max(), results_df['estimated_llc'].max())

    ax.set_xlim(left=min_llc, right=max_llc)
    ax.set_ylim(bottom=min_llc, top=max_llc)

    line = jnp.linspace(min_llc, max_llc)
    ax.plot(line, line, color='k')
    
    if title:
        ax.set_title(title)

    if figpath is None:
        plt.show()   
    else:
        plt.savefig(figpath)

    plt.close(fig)

def plot_weight_traces(weight_traces, title='', figpath=None):
    fig, ax = plt.subplots()
    for trace in weight_traces:
        w_0 = trace[:, 0]
        w_1 = trace[:, 1]
        ax.plot(w_0, w_1)

    ax.set_xlabel('w_0')
    ax.set_ylabel('w_1')
    min_weight = result['weight_traces'].min()
    max_weight = result['weight_traces'].max()

    lim_low = min(min_weight, -max_weight)
    lim_high = max(max_weight, -min_weight)

    ax.set_xlim(left=lim_low, right=lim_high)
    ax.set_ylim(bottom=lim_low, top=lim_high)

    line = jnp.linspace(lim_low, lim_high)
    ax.plot(jnp.zeros_like(line), line, color='k')
    ax.plot(line, jnp.zeros_like(line), color='k')

    if title:
        ax.set_title(title)

    if figpath is None:
        plt.show()   
    else:
        plt.savefig(figpath)
        
    plt.close(fig)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset_size", type=int, default=1000)
    parser.add_argument("--epsilon", type=float, default=0.0005)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--num_chains", type=int, default=1)
    parser.add_argument("--burn_in", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--max_power", type=int, default=4)
    

    args = parser.parse_args()

    nbeta = args.beta * args.dataset_size / jnp.log(args.dataset_size)

    batch_size = args.batch_size if args.batch_size is not None else args.dataset_size

    key = jax.random.key(args.seed)

    sampler = SGLD(args.epsilon, args.gamma, nbeta)

    os.makedirs(args.output_dir, exist_ok=True)
    exp_dir = create_unique_subdirectory(args.output_dir)

    with open(os.path.join(exp_dir, 'args.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4)

    results = []

    for powers in itertools.combinations_with_replacement(range(0, args.max_power + 1), r=2):

        if powers == (0, 0):
            continue

        k0, k1 = powers

        powers = jnp.array(powers)
        result = run_normal_crossing_experiment(key, powers, sampler, args.dataset_size, batch_size, args.num_steps, args.num_chains, args.burn_in)

        results.append({
            'k_0': powers[0].item(), 
            'k_1': powers[1].item(), 
            'estimated_llc': result['estimated_llc'], 
            'true_llc': result['true_llc'], 
        })

        # plot loss traces
        plot_loss_traces(result['loss_traces'], title=f'w0^{k0}w1^{k1}', figpath=os.path.join(exp_dir, f'loss_trace_k0={k0}_k1={k1}.png'))

        # plot weight traces
        plot_weight_traces(result['weight_traces'], title=f'w0^{k0}w1^{k1}', figpath=os.path.join(exp_dir, f'weight_trace_k0={k0}_k1={k1}.png'))

    results_df = pd.DataFrame(results)

    results_df.to_csv(os.path.join(exp_dir, 'results.csv'), index=False)
    plot_true_vs_estimated_llc(results_df, title='True vs Estimated LLC', figpath=os.path.join(exp_dir, 'true_vs_estimated_llc.png'))


    

    
