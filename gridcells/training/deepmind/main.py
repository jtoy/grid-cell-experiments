import torch
from tqdm import tqdm
from glob import glob
import datetime as dt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from gridcells.data import encoder as data_encoder
from gridcells.models import main as gridcell_models
from gridcells.data.dataset import EncodedLocationDataset
from gridcells.training.deepmind import epochs as training_epochs


def train():
    n_epochs = 21
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    date_time = dt.datetime.now().strftime("%m%d_%H%M")
    run_name = "DM_" + date_time
    writer = SummaryWriter(f"tmp/tensorboard/{run_name}")

    paths = glob('data/torch/*pt')

    encoder = data_encoder.DeepMindishEncoder()
    t_dataset = EncodedLocationDataset(paths[:50], encoder)
    v_dataset = EncodedLocationDataset(paths[50:70], encoder)

    train_loader = DataLoader(t_dataset, batch_size=1024, shuffle=True, num_workers=8)
    validation_loader = DataLoader(v_dataset, batch_size=1024, shuffle=False, num_workers=8)

    model = gridcell_models.DeepMindModel()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

    progress_bar = tqdm(range(n_epochs), total=n_epochs)
    for epoch in progress_bar:
        training_loss = training_epochs.train_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        validation_loss = 0
        validation_loss = training_epochs.validation_epoch(
            model=model,
            data_loader=validation_loader,
            device=device,
        )

        writer.add_scalar("training/loss", training_loss, epoch)
        writer.add_scalar("validation/accuracy", validation_loss, epoch)

        epoch_summary = f'Training loss: {training_loss:.2f}, validation loss: {validation_loss:.2f}'
        progress_bar.set_description(epoch_summary)

    return model
