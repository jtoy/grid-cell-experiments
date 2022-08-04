import torch
from tqdm import tqdm
from glob import glob
from torch.utils.data import DataLoader

from gridcells.models import main as gridcell_models
from gridcells.data.dataset import SelfLocationDataset
from gridcells.training import epochs as training_epochs


def main():
    paths = glob('data/torch/*pt')
    model = gridcell_models.WorstModel()
    t_dataset = SelfLocationDataset(paths[:70])
    v_dataset = SelfLocationDataset(paths[70:])

    train_loader = DataLoader(t_dataset, batch_size=1024, shuffle=True, num_workers=8)
    validation_loader = DataLoader(v_dataset, batch_size=1024, shuffle=False, num_workers=8)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

    n_epochs = 1024

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


if __name__ == '__main__':
    main()
