import numpy as np
from dataclasses import dataclass


@dataclass
class Trajectory:
    init_pos: np.array
    init_hd: np.array
    ego_vel: np.array

    target_hd: np.array
    target_pos: np.array

    def __rich_repr__(self):
        yield 'init_pos', self.init_pos


@dataclass
class TrajectoryBatch:
    init_pos: np.array
    init_hd: np.array
    ego_vel: np.array

    target_hd: np.array
    target_pos: np.array

    def __rich_repr__(self):
        yield 'init_pos', self.init_pos
        yield 'batch_size', self.batch_size

    @property
    def batch_size(self):
        return self.init_pos.shape[0]

    def __getitem__(self, idx: int) -> Trajectory:
        trajectory = Trajectory(
            init_pos=self.init_pos[idx],
            init_hd=self.init_hd[idx],
            ego_vel=self.ego_vel[idx],
            target_hd=self.target_hd[idx],
            target_pos=self.target_pos[idx],
        )
        return trajectory
