from pathlib import Path
from torch.utils.data import Dataset

from gridcells.data.structures import Trajectory
from gridcells.data import main as gridcell_data


class SelfLocationDataset(Dataset):
    def __init__(self, paths: list[Path]):
        """
            paths: have to point to our version of the data, e.g.: 'data/torch/9-99.pt'
        """
        self.paths = paths
        self.batch_trajectories = [gridcell_data.load_trajectory_batch(path) for path in paths]

    def __len__(self) -> int:
        return sum(batch.size for batch in self.batch_trajectories)

    def get_trajectory(self, idx: int) -> Trajectory:
        batch_id = idx // 10_000
        trajectory_id = idx % 10_000
        trajectory = self.batch_trajectories[batch_id][trajectory_id]

        return trajectory

    def __getitem__(self, idx: int) -> dict:
        trajectory = self.get_trajectory(idx)
        return trajectory.as_dict()


class EncodedLocationDataset(SelfLocationDataset):
    def __init__(self, paths: list[Path], encoder: None):
        super().__init__(paths=paths)
        self.encoder = encoder

    def __getitem__(self, idx: int) -> dict:
        trajectory = self.get_trajectory(idx)
        record = self.encoder.encode(trajectory)

        record |= trajectory.as_dict()
        return record
