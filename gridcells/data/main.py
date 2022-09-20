from pathlib import Path

import torch

from gridcells.data.structures import TrajectoryBatch


def load_trajectory_batch(path: Path) -> TrajectoryBatch:
    data = torch.load(path)
    batch = TrajectoryBatch(**data)
    return batch
