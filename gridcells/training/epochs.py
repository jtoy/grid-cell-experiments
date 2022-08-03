from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataclasses import dataclass


@dataclass
class LossMetric:
    n_samples: int = 0
    total_loss: float = 0.0

    @property
    def average_loss(self) -> float:
        return self.total_loss / self.n_samples


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim,
    partial_loss_fn: Callable,
) -> nn.Module:
    model.train()

    loss_metric = LossMetric()

    for batch in data_loader:
        optimizer.zero_grad()

        # Model inputs
        init_pos = batch['init_pos']
        init_hd = batch['init_hd']
        ego_vel = batch['ego_vel']

        # Model outputs
        target_place = batch['target_pos']
        target_head = batch['target_hd']

        place_cells, head_cells = model(init_pos, init_hd, ego_vel)

        place_loss = partial_loss_fn(target_place, place_cells)
        head_loss = partial_loss_fn(target_head, head_cells)
        loss = place_loss + head_loss

        loss_metric.total_loss += loss.item()
        loss_metric.n_samples += 1

        loss.backward()

        optimizer.step()

    return loss_metric.average_loss


def validation_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    partial_loss_fn: Callable,
) -> nn.Module:
    model.train()

    loss_metric = LossMetric()

    for batch in data_loader:
        # Model inputs
        init_pos = batch['init_pos']
        init_hd = batch['init_hd']
        ego_vel = batch['ego_vel']

        # Model outputs
        target_place = batch['target_pos']
        target_head = batch['target_hd']

        place_cells, head_cells = model(init_pos, init_hd, ego_vel)

        place_loss = partial_loss_fn(target_place, place_cells)
        head_loss = partial_loss_fn(target_head, head_cells)
        loss = place_loss + head_loss

        loss_metric.total_loss += loss.item()
        loss_metric.n_samples += 1

    return loss_metric.average_loss
