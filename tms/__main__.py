import os
import random
import argparse
import re
import pandas as pd
import plotille
from pathlib import Path
import tqdm
import time

import jax

from .model import TMSModel, loss
from . import plotting
from .llc import estimate_llc
from .train import train
from .data import generate_dataset

ADJECTIVES = [
        'red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray', 'black', 'white',
        'happy', 'sad', 'angry', 'calm', 'excited', 'tired', 'energetic', 'peaceful', 'nervous', 'confident',
        'brave', 'shy', 'clever', 'wise', 'foolish', 'kind', 'cruel', 'gentle', 'rough', 'smooth',
        'big', 'small', 'tall', 'short', 'long', 'wide', 'narrow', 'thick', 'thin', 'heavy', 'light',
        'fast', 'slow', 'loud', 'quiet', 'bright', 'dark', 'hot', 'cold', 'warm', 'cool',
        'old', 'new', 'young', 'ancient', 'modern', 'fresh', 'stale', 'clean', 'dirty', 'tidy', 'messy',
        'rich', 'poor', 'expensive', 'cheap', 'valuable', 'worthless', 'rare', 'common',
        'beautiful', 'ugly', 'pretty', 'handsome', 'elegant', 'plain', 'fancy', 'simple'
    ]

NOUNS = [
        'cat', 'dog', 'bird', 'fish', 'rabbit', 'horse', 'cow', 'pig', 'sheep', 'goat', 'chicken',
        'tree', 'flower', 'grass', 'bush', 'forest', 'mountain', 'hill', 'valley', 'river', 'lake', 'ocean',
        'star', 'moon', 'sun', 'planet', 'galaxy', 'universe', 'cloud', 'rain', 'snow', 'wind', 'storm',
        'book', 'pen', 'pencil', 'paper', 'notebook', 'computer', 'phone', 'tablet', 'keyboard', 'screen',
        'house', 'apartment', 'building', 'office', 'school', 'hospital', 'store', 'restaurant', 'park',
        'car', 'truck', 'bicycle', 'train', 'plane', 'boat', 'ship', 'rocket', 'submarine',
        'chair', 'table', 'bed', 'couch', 'desk', 'lamp', 'mirror', 'window', 'door', 'roof', 'floor',
        'food', 'water', 'bread', 'cheese', 'meat', 'vegetable', 'fruit', 'cake', 'cookie', 'candy',
        'music', 'art', 'movie', 'game', 'sport', 'dance', 'song', 'painting', 'sculpture', 'photograph',
        'friend', 'family', 'parent', 'child', 'baby', 'adult', 'student', 'teacher', 'doctor', 'lawyer'
    ]


def create_unique_subdirectory(base_dir, max_tries=100):
    
    base_dir = os.path.abspath(base_dir)

    for _ in range(max_tries):
        adj = random.choice(ADJECTIVES)
        noun = random.choice(NOUNS)
        new_dir_name = f"{adj}_{noun}"
        full_path = os.path.join(base_dir, new_dir_name)

        try:
            os.makedirs(full_path, exist_ok=False)
            return full_path
        except FileExistsError:
            continue
        
    print(f"Failed to generate a new subdirectory of {base_dir} after {max_tries} attempts.")
    return None

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
    step_max = steps.max().item()

    for i in tqdm.trange(len(steps), desc='Training (replay)', unit='ckpt'):
        step = steps[i]

        ckpt_file = os.path.join(run_dir, f'checkpoints/step={step}.npz')
        weights = TMSModel.load(ckpt_file)

        weights_fig_str = plotting.get_weights_figure(weights)

        llc_fig = plotille.Figure()
        llc_fig.width = 40
        llc_fig.height = 15
        llc_fig.x_label = 'Training step'
        llc_fig.y_label = 'LLC'
        llc_fig.set_x_limits(0, step_max)
        llc_fig.set_y_limits(0, llc_max)
        
        llc_fig.plot(steps[:i], llcs[:i], lc='green')

        llc_fig_str = str(llc_fig.show())

        fig_str = f"{weights_fig_str}\n{llc_fig_str}"

        if i != 0:
            fig_str = plotting.add_overwrite(fig_str)
        
        tqdm.tqdm.write(fig_str)

        time.sleep(0.2)
    

parser = argparse.ArgumentParser(description="Training script")


training_group = parser.add_argument_group("Training")

training_group.add_argument("--seed", type=int, required=True, help="Random seed")
training_group.add_argument("--num_steps", type=int, required=True, help="Number of training steps")

training_group.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
training_group.add_argument("--in_dim", type=int, default=5, help="Input dimension (default: 5)")
training_group.add_argument("--hidden_dim", type=int, default=2, help="Hidden dimension (default: 2)")
training_group.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate (default: 0.01)")
training_group.add_argument("--checkpoint_rate", type=int, default=100, help="Logging rate (default: 100)")


llc_group = parser.add_argument_group("LLC Estimation")
llc_group.add_argument('--num_draws', type=int, default=1000)
llc_group.add_argument('--num_chains', type=int, default=8)
llc_group.add_argument('--num_burnin_steps', type=int, default=1000)
llc_group.add_argument('--epsilon', type=float, default=0.0001)
llc_group.add_argument('--gamma', type=float, default=1.0)
llc_group.add_argument('--beta', type=float, default=1.0)
llc_group.add_argument('--llc_seed', type=int, default=0)





args = parser.parse_args()


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

llc_data = generate_dataset(key_llc_data, args.in_dim, args.batch_size, args.num_chains * (args.num_draws + args.num_burnin_steps))
llc_data = llc_data.reshape(args.num_chains, args.num_draws + args.num_burnin_steps, args.batch_size, args.in_dim)



for step, ckpt_file in tqdm.tqdm(sorted(get_checkpoints(checkpoint_dir)), desc='LLC estimation', unit='ckpt'):
    
    
    model = TMSModel.load(ckpt_file)

    key, key_llc_estimate = jax.random.split(key)
    llcs, _ = estimate_llc(key_llc_estimate, model, llc_data, loss, args.epsilon, args.gamma, args.beta, args.num_burnin_steps)

    
    llc_results['step'].append(step)
    llc_results['llc'].append(llcs.mean().item())

    

llc_results_df = pd.DataFrame(llc_results)
llc_results_df.to_csv(os.path.join(run_dir, 'llc.csv'))

replay_training(run_dir)




