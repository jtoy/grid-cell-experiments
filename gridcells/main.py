import torch
from tqdm import tqdm
from glob import glob
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from gridcells.models import main as gridcell_models
from gridcells.data.dataset import SelfLocationDataset
from gridcells.training import epochs as training_epochs
from gridcells.validation import views as validation_views


def main():
    n_epochs = 2048

    paths = glob('data/torch/*pt')
    model = gridcell_models.WorstModel()
    t_dataset = SelfLocationDataset(paths[:70])
    v_dataset = SelfLocationDataset(paths[70:])

    train_loader = DataLoader(t_dataset, batch_size=1024, shuffle=True, num_workers=8)
    validation_loader = DataLoader(v_dataset, batch_size=1024, shuffle=False, num_workers=8)

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
        epoch_summary = f'Training loss: {training_loss:.2f}, validation loss: {validation_loss:.2f}'
        progress_bar.set_description(epoch_summary)

    torch.save(model.state_dict(), 'tmp/model.pt')

    # Make some plots
    batch = next(iter(validation_loader))
    init_pos = batch['init_pos']
    init_hd = batch['init_hd']
    ego_vel = batch['ego_vel']
    target_place = batch['target_pos']
    target_head = batch['target_hd']
    place_cells, head_cells = model(init_pos, init_hd, ego_vel)

    n_plots = 10
    for idx in range(n_plots):
        validation_views.compare_model_output(
            init_pos=init_pos[idx].detach(),
            target_place=target_place[idx].detach(),
            model_place=place_cells[idx].detach(),
            target_head=target_head[idx].detach(),
            model_head=head_cells[idx].detach(),
        )
        plot_path = f'tmp/{idx:02}.png'
        plt.savefig(plot_path)
        plt.clf()


if __name__ == '__main__':
    main()
