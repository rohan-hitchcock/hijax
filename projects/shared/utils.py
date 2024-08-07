import os
import random
from typing import List, Type, Callable
from dataclasses import dataclass 

import jax
import numpy as np

from tqdm import tqdm
import plotille


def register_model(weights: List[str] = [], hparams: List[str] = []) -> Callable[[Type], Type]:
    """ Use as a decorator to register a custom class as a model."""
    def decorator(cls: Type) -> Type:
        cls = jax.tree_util.register_dataclass(
            cls, 
            data_fields=weights, 
            meta_fields=hparams
        )
        cls = dataclass(cls)
        return cls

    return decorator


class ExponentialMovingAverage:
    def __init__(self, decay=0.95):
        self.decay = decay
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.decay * self.value + (1 - self.decay) * new_value

    def __float__(self):
        return float(self.value) if self.value is not None else 0.0

    def __format__(self, format_spec):
        return f"{float(self):{format_spec}}"



class TrainLogger:
    def __init__(self, checkpoint_dir, checkpoint_rate: int, total_steps: int, enable_progress_bar: bool = True, enable_plot: bool = True):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_rate = checkpoint_rate
        self.ema_loss = ExponentialMovingAverage()
        self.total_steps = total_steps
        self.enable_progress_bar = enable_progress_bar
        self.enable_plot = enable_plot
        
        self.pbar = None
        self.show_plot = print
        self.losses = []

    def __enter__(self):
        self.losses = []

        if self.enable_progress_bar:
            self.pbar = tqdm(total=self.total_steps, desc="Training")
            self.show_plot = self.pbar.write
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.pbar is not None:
            self.pbar.close()
            self.show_plot = print

    def log_step(self, step: int, loss: float, model):
        self.ema_loss.update(loss)
        

        self.losses.append(loss.item())

        if self.enable_plot:
            

            fig = plotille.Figure()
            fig.width = 40
            fig.height = 15
            fig.x_label = 'Training step'
            fig.y_label = 'Loss'
            fig.set_x_limits(0, self.total_steps)
            

            losses = np.log(np.array(self.losses))
            
            fig.plot(list(range(len(self.losses))), losses, lc='green')

            fig_str = fig.show()
            if step != 0:
                fig_str = f"\x1b[{len(fig_str.splitlines())}A" + fig_str

            self.show_plot(fig_str)

        
        if self.pbar:
            self.pbar.update(1)
            self.pbar.set_postfix({"loss": f"{self.ema_loss:.4e}"})

        if self.checkpoint_dir and step % self.checkpoint_rate == 0:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'step={step}.npz')
            model.save(checkpoint_path)


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


def create_unique_subdirectory(base_dir, max_tries=100, tag=None):

    base_dir = os.path.abspath(base_dir)

    for _ in range(max_tries):
        adj = random.choice(ADJECTIVES)
        noun = random.choice(NOUNS)
        new_dir_name = f"{adj}_{noun}" if tag is None else f"{tag}_{adj}_{noun}"
        full_path = os.path.join(base_dir, new_dir_name)

        try:
            os.makedirs(full_path, exist_ok=False)
            return full_path
        except FileExistsError:
            continue

    print(f"Failed to generate a new subdirectory of {base_dir} after {max_tries} attempts.")
    return None

