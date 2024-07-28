import jax.numpy as jnp
import plotille

from itertools import zip_longest
from typing import List


from typing import Literal

import re

def line(end, start=None, samples=50):

    if start is None:
        start = jnp.zeros_like(end)

    start = start[:,jnp.newaxis]
    end = end[:,jnp.newaxis]

    alpha = jnp.linspace(0, 1, num=samples)[jnp.newaxis,:]

    return alpha * start + (1 - alpha) * end

def get_weights_figure(weights, title=''):

    fig = plotille.Figure()
    fig.width = 40
    fig.height = 15
    fig.set_x_limits(-1, 1)
    fig.set_y_limits(-1, 1)

    for col in weights.matrix.T:
        xs, ys = line(col, samples=2)
        fig.plot(xs, ys)

    figure_str = str(fig.show())
    if title:
        figure_str = f"{title}\n{figure_str}"

    return figure_str

def add_overwrite(figure_str):
    return f"\x1b[{len(figure_str.splitlines())}A" + figure_str

def stack_figures_vertically(figure_strings: List[str]):
    return '\n'.join(fs for fs in figure_strings if fs != '')

def stack_figures_horizontally(figure_strings: List[str]):

    figure_strings_line_by_line = (fig_str.splitlines(keepends=False) for fig_str in figure_strings if fig_str != '')
    
    # compute maximum line length in each figure
    figure_strings_line_by_line = ((fig_lines, max(line_width(line) for line in fig_lines)) for fig_lines in figure_strings_line_by_line)
    
    # pad each line to the length of the longest line in the figure
    figure_strings_line_by_line = [
        [pad_to_width(line, max_line_length) for line in fig_lines]
        for fig_lines, max_line_length in figure_strings_line_by_line
    ]

    supfig_lines = [' '.join(fig_lines) for fig_lines in zip_longest(*figure_strings_line_by_line, fillvalue='')]
    
    return '\n'.join(supfig_lines)


def arange_figures(figure_strings: List[List[str]]) -> str:
    return stack_figures_vertically([stack_figures_horizontally(fig_row) for fig_row in figure_strings])



def pad_to_width(string: str, length: int, padding: str = ' ', padding_mode: Literal['left', 'right'] = 'right') -> str:

    string_width = line_width(string)
    num_padding = max(length - string_width, 0)

    padding_string = num_padding * padding 
    return f"{padding_string}{string}" if padding_mode == 'left' else f"{string}{padding_string}"

def strip_formatting(text):
    # currently only handles basic color formatting
    color_escape = re.compile(r'\x1B\[\d+m')
    return color_escape.sub('', text)

def line_width(line):
    return len(strip_formatting(line))
