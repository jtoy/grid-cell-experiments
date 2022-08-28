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
) -> nn.Module:
    model.train()

    loss_metric = LossMetric()

    for batch in tqdm(data_loader):
        optimizer.zero_grad()

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

        pc_loss = F.cross_entropy(predicted_positions.view(-1, 256), target_pos.argmax(2).view(-1))
        hd_loss = F.cross_entropy(predicted_hd.view(-1, 12), target_hd.argmax(2).view(-1))

        loss = (pc_loss + hd_loss) / 2

        loss_metric.total_loss += loss.item()
        # MSE is already per-sample
        loss_metric.n_samples += 1

        loss.backward()

        optimizer.step()

    return loss_metric.average_loss
