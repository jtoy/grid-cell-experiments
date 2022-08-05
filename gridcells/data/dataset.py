from pathlib import Path
from torch.utils.data import Dataset

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

    def __getitem__(self, idx: int) -> dict:
        batch_id = idx // 10_000
        trajectory_id = idx % 10_000
        trajectory = self.batch_trajectories[batch_id][trajectory_id]
        return trajectory.as_dict()
