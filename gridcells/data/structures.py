import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class Trajectory:
    init_pos: np.array
    init_hd: np.array
    ego_vel: np.array

    target_hd: np.array
    target_pos: np.array

    def as_dict(self) -> dict:
        return asdict(self)

    def __rich_repr__(self):
        yield 'init_pos', self.init_pos

    def recreate_trajectory(self) -> np.array:
        """
        Paper description of the *ego_vel* property:

        In the supervised setup the grid cell network receives,
        at each step t, the egocentric linear velocity vt ∈ R and the sine and cosine of its
        angular velocity ϕt.
        """
        # Initial conditions
        position = self.init_pos
        head_direction = self.init_hd[0]

        positions = []
        head_directions = []
        for ego_vel in self.ego_vel:
            # Position step
            egocentric_linear_velocity = ego_vel[0]
            dx = np.cos(head_direction) * egocentric_linear_velocity
            dy = np.sin(head_direction) * egocentric_linear_velocity
            new_x = position[0] + dx
            new_y = position[1] + dy
            position = np.array([new_x, new_y])
            positions.append(position)

            # Head direction step
            angular_velocity_cos = ego_vel[1]
            angular_velocity_sine = ego_vel[2]
            # Head direction is an angle, but they store it as sine and cosine
            dhead_direction = np.arctan2(angular_velocity_sine, angular_velocity_cos)
            head_direction = head_direction + dhead_direction
            head_directions.append(head_direction)

        positions = np.array(positions)
        return positions


@dataclass
class TrajectoryBatch:
    init_pos: np.array
    init_hd: np.array
    ego_vel: np.array

    target_hd: np.array
    target_pos: np.array

    @property
    def size(self) -> int:
        return self.init_pos.shape[0]

    def __rich_repr__(self):
        yield 'init_pos', self.init_pos
        yield 'batch_size', self.size

    def __getitem__(self, idx: int) -> Trajectory:
        trajectory = Trajectory(
            init_pos=self.init_pos[idx],
            init_hd=self.init_hd[idx],
            ego_vel=self.ego_vel[idx],
            target_hd=self.target_hd[idx],
            target_pos=self.target_pos[idx],
        )
        return trajectory
