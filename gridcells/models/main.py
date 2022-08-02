import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class WorstModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(203, 100)
        self.fc2 = nn.Linear(100, 100)
        self.out_place = nn.Linear(100, 200)
        self.out_head = nn.Linear(100, 100)

    def forward(self, init_pos: Tensor, init_hd: Tensor, ego_vel: Tensor) -> tuple[Tensor, Tensor]:
        head_rotation_speed = torch.arctan2(ego_vel[..., 1], ego_vel[..., 2])
        forward_speed = ego_vel[..., 0]
        x = torch.hstack([head_rotation_speed, forward_speed, init_hd, init_pos])

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        place_cells = self.out_place(x)
        place_cells = place_cells.view(-1, 100, 2)
        head_cells = self.out_head(x)
        return place_cells, head_cells
