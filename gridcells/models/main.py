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


class DeepMindModel(nn.Module):
    def __init__(self, weight_decay: float = 1e-5):
        super().__init__()

        self.weight_decay = weight_decay

        self.l1 = nn.Linear(1044, 128)
        self.l2 = nn.Linear(1044, 128)
        self.rnn = nn.LSTMCell(input_size=3, hidden_size=128)
        self.bottleneck_layer = nn.Linear(128, 256, bias=False)
        self.pc_logits = nn.Linear(256, 256)
        self.hd_logits = nn.Linear(256, 12)

    @property
    def number_of_parameters(self) -> int:
        how_many = sum(p.numel() for p in self.parameters())
        return how_many

    def forward(self, concat_init, ego_vel):
        init_lstm_cell = self.l1(concat_init)
        init_lstm_state = self.l2(concat_init)
        (hx, cx) = (init_lstm_state, init_lstm_cell)

        lstm_inputs = ego_vel.transpose(0, 1)

        bottlenecks = []
        predicted_positions = []
        predicted_hd = []
        for lstm_input in lstm_inputs:
            hx, cx = self.rnn(lstm_input, (hx, cx))

            bottleneck = self.bottleneck_layer(hx)
            bottleneck = nn.functional.dropout(bottleneck, 0.5)
            bottlenecks.append(bottleneck)

            predicted_position = self.pc_logits(bottleneck)
            predicted_positions.append(predicted_position)

            predicted_head_direction = self.hd_logits(bottleneck)
            predicted_hd.append(predicted_head_direction)

        bottlenecks = torch.stack(bottlenecks, dim=1)
        predicted_positions = torch.stack(predicted_positions, dim=1)
        predicted_hd = torch.stack(predicted_hd, dim=1)

        return predicted_positions, predicted_hd, bottlenecks

    @property
    def l2_bottleneck(self):
        loss = self.bottleneck_layer.weight.norm(2) + self.pc_logits.weight.norm(2) + self.hd_logits.weight.norm(2)
        return loss

    def regularization(self) -> torch.Tensor:
        loss = self.weight_decay * self.l2_bottleneck
        return loss
