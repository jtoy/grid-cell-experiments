from matplotlib import pyplot as plt

from gridcells.data.structures import Trajectory


def draw_trajectory(trajectory: Trajectory):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    x0 = trajectory.init_pos[0]
    y0 = trajectory.init_pos[1]
    ax.plot(x0, y0, 'o')

    x = trajectory.target_pos[:, 0]
    y = trajectory.target_pos[:, 1]

    ax.plot(x, y, '--o')

    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(-1.1, 1.1)

    return fig
