import torch
import torch.nn as nn
from torch.nn import functional as F
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
    device: torch.device,
) -> float:
    model.train()

    loss_metric = LossMetric()

    for batch in data_loader:
        optimizer.zero_grad()

        loss = process(model, batch, device)

        loss_metric.total_loss += loss.item()
        loss_metric.n_samples += 1

        loss.backward()

        optimizer.step()

    return loss_metric.average_loss


def validation_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()

    loss_metric = LossMetric()

    for batch in data_loader:
        loss = process(model, batch, device)

        loss_metric.total_loss += loss.item()
        loss_metric.n_samples += 1

    return loss_metric.average_loss


def process(
    model: nn.Module,
    batch: dict,
    device: torch.device,
) -> torch.tensor:
    # Model inputs
    ego_vel = batch['ego_vel'].float().to(device)
    encoded_inits = batch['encoded_inits']
    encoded_pos = encoded_inits['position'].float().to(device)
    encoded_hd = encoded_inits['head_direction'].float().to(device)
    concat_init = torch.cat([encoded_hd, encoded_pos], axis=2).squeeze()

    # Model outputs
    target_pos = batch['encoded_targets']['position'].float().to(device)
    target_hd = batch['encoded_targets']['head_direction'].float().to(device)

    predicted_positions, predicted_hd, bottlenecks = model(concat_init, ego_vel)

    # TODO: Remove hardcoded values
    pc_loss = F.cross_entropy(predicted_positions.view(-1, 256), target_pos.argmax(2).view(-1))
    hd_loss = F.cross_entropy(predicted_hd.view(-1, 12), target_hd.argmax(2).view(-1))

    loss = (pc_loss + hd_loss) / 2

    return loss
