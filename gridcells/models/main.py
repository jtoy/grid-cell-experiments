import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class WorstModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(203, 100)
        self.fc2 = nn.Linear(100, 100)
        self.out_place_cells = nn.Linear(100, 200)
        self.out_head_cells = nn.Linear(100, 100)

    def forward(self, init_pos: Tensor, init_hd: Tensor, ego_vel: Tensor) -> tuple[Tensor, Tensor]:
        # Concatenate everything into a single vector
        head_rotation_speed = torch.arctan2(ego_vel[..., 1], ego_vel[..., 2])
        forward_speed = ego_vel[..., 0]
        x = torch.hstack([head_rotation_speed, forward_speed, init_hd, init_pos])

        # Run it through a MLP
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Decode the output into two sets of "cells"
        # Expected shape for position
        place_cells = self.out_place_cells(x)
        place_cells = place_cells.view(-1, 100, 2)

        # Expected shape for head direction
        head_cells = self.out_head_cells(x)
        head_cells = head_cells.view(-1, 100, 1)
        return place_cells, head_cells
