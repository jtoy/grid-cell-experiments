import torch
from tqdm import tqdm
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

    batches_per_epoch = 1000
    # progress_bar = tqdm(data_loader, total=len(data_loader), leave=False)
    progress_bar = tqdm(enumerate(data_loader), total=batches_per_epoch, leave=False)
    # for batch in data_loader:
    for it, batch in progress_bar:
        optimizer.zero_grad()

        loss = process(model, batch, device)

        loss_metric.total_loss += loss.item()
        loss_metric.n_samples += 1

        loss.backward()

        clipping_value = 1e-5
        torch.nn.utils.clip_grad_value_(model.parameters(), clipping_value)

        optimizer.step()
        desc = f'Training Loss: {loss.item():.2f}'
        progress_bar.set_description(desc)
        if it >= batches_per_epoch:
            break

    return loss_metric.average_loss


def validation_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()

    loss_metric = LossMetric()

    # for batch in data_loader:
    progress_bar = tqdm(data_loader, total=len(data_loader), leave=False)
    for batch in progress_bar:
        loss = process(model, batch, device)

        loss_metric.total_loss += loss.item()
        loss_metric.n_samples += 1

        desc = f'Validation Loss: {loss.item():.2f}'
        progress_bar.set_description(desc)

    return loss_metric.average_loss


def process(
    model: nn.Module,
    batch: dict,
    device: torch.device,
) -> torch.tensor:
    # Model inputs
    ego_vel = batch['ego_vel'].to(device, dtype=torch.float32)
    encoded_pos = batch['encoded_initial_pos'].to(device, dtype=torch.float32)
    encoded_hd = batch['encoded_initial_hd'].to(device, dtype=torch.float32)
    concat_init = torch.cat([encoded_hd, encoded_pos], axis=2).squeeze()

    # Model outputs
    target_pos = batch['encoded_target_pos'].to(device, dtype=torch.float32)
    target_hd = batch['encoded_target_hd'].to(device, dtype=torch.float32)

    predicted_positions, predicted_hd, bottlenecks = model(concat_init, ego_vel)

    # TODO: Remove hardcoded values
    pc_loss = F.cross_entropy(predicted_positions.view(-1, 256), target_pos.argmax(2).view(-1))
    hd_loss = F.cross_entropy(predicted_hd.view(-1, 12), target_hd.argmax(2).view(-1))

    loss = pc_loss + hd_loss + model.regularization()

    return loss
