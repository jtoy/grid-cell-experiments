import torch
import numpy as np
from tqdm import tqdm
from glob import glob
import datetime as dt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from gridcells.data import encoder as data_encoder
from gridcells.models import main as gridcell_models
from gridcells.data.dataset import CachedEncodedDataset
from gridcells.validation import views as validation_views
from gridcells.training.deepmind import epochs as training_epochs
from gridcells.training.base.rmsprop_tf import RMSprop as RMSprop_tf


def train():
    n_epochs = 201
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    date_time = dt.datetime.now().strftime("%m%d_%H%M")
    run_name = "DM_" + date_time
    writer = SummaryWriter(f"tmp/tensorboard/{run_name}")

    paths = glob('data/encoded_pickles/*pickle')

    t_dataset = CachedEncodedDataset(paths[:30])
    v_dataset = CachedEncodedDataset(paths[30:40])

    batch_size = 2500
    num_workers = 4
    train_loader = DataLoader(
        t_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        v_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = gridcell_models.DeepMindModel()
    model = model.to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
    optimizer = RMSprop_tf(model.parameters(), lr=0.001)


    progress_bar = tqdm(range(n_epochs), total=n_epochs)
    for epoch in progress_bar:
        training_loss = training_epochs.train_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        validation_loss = training_epochs.validation_epoch(
            model=model,
            data_loader=validation_loader,
            device=device,
        )

        writer.add_scalar("training/loss", training_loss, epoch)
        writer.add_scalar("validation/accuracy", validation_loss, epoch)

        epoch_summary = f'Training loss: {training_loss:.2f}, validation loss: {validation_loss:.2f}'
        progress_bar.set_description(epoch_summary)

    torch.save(model.state_dict(), 'tmp/dp-model.pt')

    return model


def review(model_state_path: str):
    # Run tests on cpu
    device = torch.device("cpu")

    # Draw this many charts
    n_samples = 10

    model = gridcell_models.DeepMindModel()
    model_state = torch.load(model_state_path)
    model.load_state_dict(model_state)
    model.cpu().eval()

    # Make a dataset from a single path that was
    # not used during trainig and validation
    paths = glob('data/encoded_pickles/*pickle')
    path = paths[-2]
    dataset = CachedEncodedDataset([path])
    loader = DataLoader(dataset, batch_size=n_samples, shuffle=True)

    hd_encoder = data_encoder.DeepMindHeadEncoder()
    position_encoder = data_encoder.DeepMindPlaceEncoder()

    # Unpack data for model input
    batch = next(iter(loader))
    ego_vel = batch['ego_vel'].to(device)
    encoded_pos = batch['encoded_initial_pos'].to(device)
    encoded_hd = batch['encoded_initial_hd'].to(device)
    concat_init = torch.cat([encoded_hd, encoded_pos], axis=2).squeeze()
    # And for drawing charts
    init_pos = batch['init_pos'].detach().numpy()
    target_hd = batch['target_hd'].numpy()
    target_pos = batch['target_pos'].numpy()

    predicted_positions, predicted_hd, bottlenecks = model(concat_init, ego_vel)
    predicted_positions = predicted_positions.detach().numpy()
    predicted_hd = predicted_hd.detach().numpy()

    for it in range(n_samples):
        decoded_hd = hd_encoder.decode(predicted_hd[it])
        # There's a shape incosistency between deepmind pipeline
        # and the base pipeline, so I need to add a dimension here to re-use plots
        decoded_hd = decoded_hd[:, np.newaxis]
        decoded_position = position_encoder.decode(predicted_positions[it])

        fig = validation_views.compare_model_output(
            init_pos=init_pos[it],
            target_place=target_pos[it],
            model_place=decoded_position,
            target_head=target_hd[it],
            model_head=decoded_hd,
        )
        savepath = f'tmp/dp-review-{it:02}.png'
        fig.savefig(savepath)
