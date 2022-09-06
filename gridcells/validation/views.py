import numpy as np
from matplotlib import pyplot as plt

from gridcells.validation import sac as SAC


def compare_model_output(
    init_pos: np.array,
    target_place: np.array,
    model_place: np.array,
    target_head: np.array,
    model_head: np.array,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={'height_ratios': [3, 1]})
    ax = axes[0]

    model_x = model_place[:, 0]
    model_y = model_place[:, 1]
    ax.plot(model_x, model_y, '--o', label='Prediction', alpha=0.666)

    real_x = target_place[:, 0]
    real_y = target_place[:, 1]
    ax.plot(real_x, real_y, '--o', label='Reality', alpha=0.666)

    x0 = init_pos[0]
    y0 = init_pos[1]
    ax.plot(x0, y0, 'o', label='Starting point')

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

    return fig


def review_position_encoder(positions: np.array, encoder):
    encoded = encoder.encode(positions)
    decoded = encoder.decode(encoded)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(positions[:, 0], positions[:, 1], label='trajectory')
    # ax.scatter(decoded[:, 0], decoded[:, 1], label='decoded')
    # ax.plot(positions[:, 0], positions[:, 1])
    ax.plot(decoded[:, 0], decoded[:, 1], '-o', label='decoded', color='tomato')

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    ax.legend()


def draw_activations_ratemap(
    xs: np.array,
    ys: np.array,
    activations: np.array,
) -> plt.Figure:
    f, axes = plt.subplots(
        nrows=16,
        ncols=16,
        figsize=[12, 12],
        gridspec_kw={
            'hspace': 0,
            'wspace': 0
        },
    )
    for it in range(16):
        for jt in range(16):
            kt = it * 16 + jt
            ax = axes[it][jt]
            ratemap = SAC.calculate_ratemap(xs, ys, activations[:, kt])
            ax.imshow(ratemap, interpolation=None, cmap='jet')
            ax.tick_params(
                axis='both',
                which='both',
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False
            )
    f.tight_layout()

    return f
