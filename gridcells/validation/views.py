import numpy as np
from matplotlib import pyplot as plt


def compare_model_output(
    init_pos: np.array,
    target_place: np.array,
    model_place: np.array,
    target_head: np.array,
    model_head: np.array,
):
    fig, axes = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={'height_ratios': [3, 1]})

    x0 = init_pos[0]
    y0 = init_pos[1]
    ax = axes[0]
    ax.plot(x0, y0, 'o')

    model_x = model_place[:, 0]
    model_y = model_place[:, 1]
    ax.plot(model_x, model_y, '--o', label='Prediction')

    real_x = target_place[:, 0]
    real_y = target_place[:, 1]
    ax.plot(real_x, real_y, '--o', label='Reality')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    ax.grid()
    ax.set_title('Position')

    ax = axes[1]
    ax.plot(model_head[:, 0], '--o', label='Prediction')
    ax.plot(target_head[:, 0], '--o', label='Reality')
    ax.legend()
    ax.set_title('Head direction')
    fig.tight_layout()
