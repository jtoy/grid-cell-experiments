import torch
from tqdm import tqdm
from glob import glob
import datetime as dt
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from gridcells.models import main as gridcell_models
from gridcells.data.dataset import SelfLocationDataset
from gridcells.validation import views as validation_views
from gridcells.training.base import epochs as training_epochs


def train():
    n_epochs = 51

    date_time = dt.datetime.now().strftime("%m%d_%H%M")
    run_name = "GC_" + date_time
    writer = SummaryWriter(f"tmp/tensorboard/{run_name}")

    paths = glob('data/torch/*pt')
    model = gridcell_models.WorstModel()
    t_dataset = SelfLocationDataset(paths[:70])
    v_dataset = SelfLocationDataset(paths[70:])

    train_loader = DataLoader(t_dataset, batch_size=1024, shuffle=True, num_workers=8)
    validation_loader = DataLoader(v_dataset, batch_size=1024, shuffle=False, num_workers=8)

    # Make some plots
    validation_batch = next(iter(validation_loader))

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

    progress_bar = tqdm(range(n_epochs), total=n_epochs)
    for epoch in progress_bar:
        training_loss = training_epochs.train_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            partial_loss_fn=loss_fn,
        )

        validation_loss = training_epochs.validation_epoch(
            model=model,
            data_loader=validation_loader,
            partial_loss_fn=loss_fn,
        )

        writer.add_scalar("training/loss", training_loss, epoch)
        writer.add_scalar("validation/accuracy", validation_loss, epoch)

        epoch_summary = f'Training loss: {training_loss:.2f}, validation loss: {validation_loss:.2f}'
        progress_bar.set_description(epoch_summary)

        if epoch % 10 == 0:
            write_validation_plots(
                model=model,
                batch=validation_batch,
                writer=writer,
                epoch=epoch,
            )

    torch.save(model.state_dict(), 'tmp/model.pt')


def write_validation_plots(
    model: nn.Module,
    batch: dict,
    writer: SummaryWriter,
    epoch: int,
    n_samples: int = 5
):
    init_pos = batch['init_pos']
    init_hd = batch['init_hd']
    ego_vel = batch['ego_vel']
    target_place = batch['target_pos']
    target_head = batch['target_hd']
    place_cells, head_cells = model(init_pos, init_hd, ego_vel)
    for idx in range(n_samples):
        fig = validation_views.compare_model_output(
            init_pos=init_pos[idx].detach(),
            target_place=target_place[idx].detach(),
            model_place=place_cells[idx].detach(),
            target_head=target_head[idx].detach(),
            model_head=head_cells[idx].detach(),
        )
        writer.add_figure(f"validation/trajectories/{idx:02}", fig, epoch)
