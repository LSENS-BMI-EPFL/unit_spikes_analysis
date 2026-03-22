import os
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_array(x):
    """Convert string representation of array to numpy array."""
    if isinstance(x, str):
        x = x.replace('\n', ' ').replace('  ', ' ')
        x = x.replace('[', '').replace(']', '')
        if x.strip() == '':
            return np.array([])
        return np.array([float(v) for v in x.split() if v != ''])
    return x

def save_figure_multiple_formats(fig, filepath_base, dpi=300):
    """Save matplotlib figure as PNG, PDF, and SVG files."""
    dir_name = os.path.dirname(filepath_base)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    for ext in ['png', 'pdf', 'svg']:
        filepath = f"{filepath_base}.{ext}"
        if ext == 'svg':
            fig.savefig(filepath, format='svg', bbox_inches='tight')
        else:
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

