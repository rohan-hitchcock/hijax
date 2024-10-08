import os
import argparse
import re
import pandas as pd
import plotille
from pathlib import Path
import tqdm
import time

import jax
import jax.numpy as jnp

from shared.utils import create_unique_subdirectory 

from .model import TMSModel, loss_fn
from . import plotting
from .llc import estimate_llc, llc_sweep
from .train import train
from .data import generate_dataset


from dln import llc as llc_dln
from shared.samplers_optax import SGLD

def get_checkpoints(directory):
    
    checkpoint_pattern = re.compile(r'step=(\d+)\.npz')
    checkpoint_files = []

    directory_path = Path(directory)

    for filepath in directory_path.iterdir():
        if filepath.is_file():
            match = checkpoint_pattern.match(filepath.name)
            if match:
                step = int(match.group(1))
                checkpoint_files.append((step, str(filepath)))

    return checkpoint_files

def replay_training(run_dir):

    llc_df = pd.read_csv(os.path.join(run_dir, 'llc.csv'))

    steps = llc_df['step'].to_numpy()
    llcs = llc_df['llc'].to_numpy()

    llc_max = llcs.max().item()
    llc_min = llcs.min().item()
    step_max = steps.max().item()

    num_ckpts = len(steps)
    next_weight_freeze = num_ckpts // 3
    weight_history_fig = ''

    for i in tqdm.trange(len(steps), desc='Training (replay)', unit='ckpt'):
        step = steps[i]

        ckpt_file = os.path.join(run_dir, f'checkpoints/step={step}.npz')
        weights = TMSModel.load(ckpt_file)

        weights_fig_str = plotting.get_weights_figure(weights, title=f"Step {step}")

        if i == next_weight_freeze + 1:
            weight_history_fig = plotting.stack_figures_horizontally([weight_history_fig, weights_fig_str])
            next_weight_freeze += num_ckpts // 3


        llc_fig = plotille.Figure()
        llc_fig.width = 180
        llc_fig.height = 15
        llc_fig.x_label = 'Training step'
        llc_fig.y_label = 'LLC'
        llc_fig.set_x_limits(0, step_max)
        llc_fig.set_y_limits(llc_min, llc_max)
        
        llc_fig.plot(steps[:i], llcs[:i], lc='green')

        llc_fig_str = str(llc_fig.show())

        
        fig_str = plotting.arrange_figures(
            [
                [weight_history_fig, weights_fig_str],
                [llc_fig_str]
            ]
        )

        if i != 0:
            fig_str = plotting.add_overwrite(fig_str)
        
        tqdm.tqdm.write(fig_str)
        
        time.sleep(0.1)
    

parser = argparse.ArgumentParser(description="Training script")


training_group = parser.add_argument_group("Training")

training_group.add_argument("--seed", type=int, required=True, help="Random seed")
training_group.add_argument("--num_steps", type=int, required=True, help="Number of training steps")

training_group.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
training_group.add_argument("--in_dim", type=int, default=6, help="Input dimension (default: 5)")
training_group.add_argument("--hidden_dim", type=int, default=2, help="Hidden dimension (default: 2)")
training_group.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate (default: 0.01)")
training_group.add_argument("--checkpoint_rate", type=int, default=100, help="Logging rate (default: 100)")


llc_group = parser.add_argument_group("LLC Estimation")
llc_group.add_argument('--num_draws', type=int, default=500)
llc_group.add_argument('--num_chains', type=int, default=20)
llc_group.add_argument('--num_burnin_steps', type=int, default=5500)
llc_group.add_argument('--epsilon', type=float, default=0.001)
llc_group.add_argument('--gamma', type=float, default=10.0)
llc_group.add_argument('--beta', type=float, default=1.0)
llc_group.add_argument('--llc_seed', type=int, default=0)

sweep_group = parser.add_argument_group("LLC Sweep")
sweep_group.add_argument('--sweep', action='store_true', help='Pass this flag to do an LLC hyperparameter sweep instead.')
sweep_group.add_argument('--num_checkpoints', type=int, default=1)
sweep_group.add_argument('--sweep_chains', type=int, default=4)
sweep_group.add_argument('--llc_batch_size', type=int, default=256)
sweep_group.add_argument('--epsilons', nargs='+', default=[0.00001, 0.0001, 0.001])
sweep_group.add_argument('--gammas', nargs='+', default=[1, 10, 100])
sweep_group.add_argument('--betas', nargs='+', default=[1, 10, 100])

args = parser.parse_args()

os.makedirs('./tms/runs', exist_ok=True)
run_dir = create_unique_subdirectory('./tms/runs')
checkpoint_dir = os.path.join(run_dir, 'checkpoints')
os.makedirs(checkpoint_dir)


key = jax.random.key(args.seed)
key, train_key = jax.random.split(key)

train(
    key=train_key, 
    num_steps=args.num_steps, 
    batch_size=args.batch_size, 
    in_dim=args.in_dim, 
    hidden_dim=args.hidden_dim, 
    learning_rate=args.learning_rate, 
    checkpoint_rate=args.checkpoint_rate, 
    checkpoint_dir=checkpoint_dir
)

llc_results = {
    'step': [], 
    'llc': []
}

key, key_llc_data = jax.random.split(key)



if args.sweep:

    sweep_draws = args.num_draws + args.num_burnin_steps

    llc_data = generate_dataset(key_llc_data, args.in_dim, args.llc_batch_size, args.sweep_chains * sweep_draws)
    llc_data = llc_data.reshape(args.sweep_chains, sweep_draws, args.llc_batch_size, args.in_dim)

    checkpoints = get_checkpoints(checkpoint_dir)
    checkpoints.sort()

    ckpt_ids_to_sweep = ( int ( round ((i / args.num_checkpoints) * (len(checkpoints) - 1) )) for i in range(1, args.num_checkpoints + 1))
    models = [(checkpoints[ckpt_id][0], TMSModel.load(checkpoints[ckpt_id][1])) for ckpt_id in ckpt_ids_to_sweep]

    for step, model in models:
        key, key_sweep = jax.random.split(key)

        epsilons = jnp.array(args.epsilons)
        gammas = jnp.array(args.gammas)
        betas = jnp.array(args.betas)

        sweep_results = llc_sweep(key_sweep, model, llc_data, loss_fn, epsilons, gammas, betas)
        
        for b, fig in zip(betas, plotting.plot_sweep(sweep_results, epsilons, gammas, betas)):
            print(f" Step = {step}, Beta = {b}".rjust(180, '*'))
            print()
            print(fig)
            print()

    exit()


# LLC_MODE = 'old'
LLC_MODE = 'new'

llc_data = generate_dataset(key_llc_data, args.in_dim, args.batch_size, args.num_chains * (args.num_draws + args.num_burnin_steps))
llc_data = llc_data.reshape(args.num_chains, args.num_draws + args.num_burnin_steps, args.batch_size, args.in_dim)

if LLC_MODE == 'new':
    sample_llc_multichain = jax.jit(jax.vmap(llc_dln.sample_llc, in_axes=(0, None, None, 0)), static_argnames=['sampler'])
    nbeta = llc_dln.nbeta(args.beta, args.batch_size)
    sampler = SGLD(args.epsilon, args.gamma, nbeta, args.seed)


for step, ckpt_file in tqdm.tqdm(sorted(get_checkpoints(checkpoint_dir)), desc='LLC estimation', unit='ckpt'):
     
    model = TMSModel.load(ckpt_file)

    key, key_llc_estimate = jax.random.split(key)


    if LLC_MODE == 'new':
        key_chains = jax.random.split(key_llc_estimate, args.num_chains)

        
        traces = sample_llc_multichain(key_chains, model, sampler, llc_data)
        llcs = jax.vmap(llc_dln.estimate_llc, in_axes=(0, None, None))(traces, args.num_burnin_steps, nbeta)

    else:
        llcs, traces = estimate_llc(key_llc_estimate, model, llc_data, loss_fn, args.epsilon, args.gamma, args.beta, args.num_burnin_steps)

    llc_results['step'].append(step)
    llc_results['llc'].append(llcs.mean().item())


import matplotlib.pyplot as plt

fig, ax = plt.subplots()
for trace in traces:
    ax.plot(trace)
plt.show()

llc_results_df = pd.DataFrame(llc_results)
llc_results_df.to_csv(os.path.join(run_dir, 'llc.csv'))

replay_training(run_dir)
