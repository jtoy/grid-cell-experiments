import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from glob import glob
import datetime as dt
from dataclasses import dataclass
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from gridcells.validation import sac as SAC
from gridcells.data import encoder as data_encoder
from gridcells.models import main as gridcell_models
from gridcells.data.dataset import CachedEncodedDataset
from gridcells.validation import views as validation_views
from gridcells.training.deepmind import epochs as training_epochs
from gridcells.training.base.rmsprop_tf import RMSprop as RMSprop_tf


@dataclass
class Config:
    batch_size: int = 10
    n_epochs: int = 301

    weight_decay: float = 1e-5


def train():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    date_time = dt.datetime.now().strftime("%m%d_%H%M")
    run_name = "DM_" + date_time
    writer = SummaryWriter(f"tmp/tensorboard/{run_name}")

    paths = glob('data/encoded_pickles/*pickle')

    t_dataset = CachedEncodedDataset(paths[:5])
    v_dataset = CachedEncodedDataset(paths[30:35])
    test_batch = make_test_batch(paths[44])

    num_workers = 8
    train_loader = DataLoader(
        t_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        v_dataset,
        batch_size=2500,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = gridcell_models.DeepMindModel(config.weight_decay)
    model = model.to(device)

    # Default value in tensorflow is different than default value in torch
    # it's also called *rho* rather than *alpha* (I think)
    alpha = 0.9
    momentum = 0.9
    learning_rate = 1e-4
    eps = 1e-10
    optimizer = RMSprop_tf(
        params=model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        alpha=alpha,
        eps=eps,
    )
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)


    progress_bar = tqdm(range(config.n_epochs), total=config.n_epochs)
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

        epoch_summary = f'Training: {training_loss:.2f}, validation: {validation_loss:.2f}'
        progress_bar.set_description(epoch_summary)

        # Detailed validation
        if epoch % 2 == 0:
            save_experiment(model, optimizer, config, run_name)
            ratemaps = make_scored_ratemaps(model, device, test_batch)

            # Passing a 'sort key' as an argument to the view might be cleaner
            ratemaps = sorted(ratemaps, key=lambda r: -r.s60)
            fig = validation_views.draw_rated_activations_ratemaps(
                ratemaps=ratemaps,
                scores=[r.s60 for r in ratemaps],
            )
            writer.add_figure("validation/s60_ratemaps", fig, epoch)

            ratemaps = sorted(ratemaps, key=lambda r: -r.s90)
            fig = validation_views.draw_rated_activations_ratemaps(
                ratemaps=ratemaps,
                scores=[r.s90 for r in ratemaps],
            )
            writer.add_figure("validation/s90_ratemaps", fig, epoch)
            review_path_integration_batch(model,test_batch,device,writer,epoch)

    save_experiment(model, optimizer, config, run_name)

    return model


def save_experiment(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Config,
    run_name: str,
):
    experiment_state = {
        'config': config,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
    }
    torch.save(experiment_state, f'tmp/{run_name}.pt')


def review_path_integration_batch(model:nn.Module, batch:dict,device:str,writer,epoch:int):
    # Draw this many charts
    n_samples = 10
    ego_vel = batch['ego_vel'].to(device)
    encoded_pos = batch['encoded_initial_pos'].to(device)
    encoded_hd = batch['encoded_initial_hd'].to(device)
    concat_init = torch.cat([encoded_hd, encoded_pos], axis=2).squeeze()
    # And for drawing charts
    init_pos = batch['init_pos'].detach().numpy()
    target_hd = batch['target_hd'].numpy()
    target_pos = batch['target_pos'].numpy()

    predicted_positions, predicted_hd, bottlenecks = model(concat_init, ego_vel)
    predicted_positions = predicted_positions.cpu().detach().numpy()
    predicted_hd = predicted_hd.cpu().detach().numpy()
    hd_encoder = data_encoder.DeepMindHeadEncoder()
    position_encoder = data_encoder.DeepMindPlaceEncoder()

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
        image = Image.open(savepath)
        transform = transforms.Compose([ transforms.PILToTensor() ])
        img_tensor = transform(image)
        writer.add_image("path_integration",img_tensor,epoch)


def review_path_integration(model_state_path: str):
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
        return savepath


def make_test_batch(path: str, n_samples: int = 5000) -> dict:
    dataset = CachedEncodedDataset([path])
    loader = DataLoader(dataset, batch_size=n_samples, shuffle=True)

    batch = next(iter(loader))

    return batch


def make_scored_ratemaps(model: nn.Module, device: torch.device, batch: dict) -> list[SAC.Ratemap]:
    ego_vel = batch['ego_vel'].to(device)
    encoded_pos = batch['encoded_initial_pos'].to(device)
    encoded_hd = batch['encoded_initial_hd'].to(device)
    concat_init = torch.cat([encoded_hd, encoded_pos], axis=2).squeeze()
    target_pos = batch['target_pos'].numpy()

    model.eval()
    predicted_positions, predicted_hd, bottlenecks = model(concat_init, ego_vel)

    data_xy = target_pos.reshape(-1, target_pos.shape[-1])
    x = data_xy[:, 0]
    y = data_xy[:, 1]

    activations = bottlenecks.reshape(-1, 256).detach().cpu().numpy()
    ratemaps = [SAC.calculate_ratemap(x, y, activations[:, kt]) for kt in range(256)]

    scorer = SAC.GridScorer()
    rated_maps = [scorer.get_grid_score(r) for r in ratemaps]

    return rated_maps


def draw_ratemaps(model: nn.Module, device: torch.device, batch: dict):
    ego_vel = batch['ego_vel'].to(device)
    encoded_pos = batch['encoded_initial_pos'].to(device)
    encoded_hd = batch['encoded_initial_hd'].to(device)
    concat_init = torch.cat([encoded_hd, encoded_pos], axis=2).squeeze()
    target_pos = batch['target_pos'].numpy()

    model.eval()
    predicted_positions, predicted_hd, bottlenecks = model(concat_init, ego_vel)

    data_xy = target_pos.reshape(-1, target_pos.shape[-1])
    x = data_xy[:, 0]
    y = data_xy[:, 1]

    activations = bottlenecks.reshape(-1, 256).detach().cpu().numpy()
    ratemaps = [SAC.calculate_ratemap(x, y, activations[:, kt]) for kt in range(256)]

    scorer = SAC.GridScorer()
    rated_maps = [scorer.get_grid_score(r) for r in ratemaps]

    # Sort by the hexagonal grid factor, descending
    ratemaps = sorted(rated_maps, key=lambda r: -r.s60)

    fig = validation_views.draw_rated_activations_ratemaps(ratemaps)
    return fig


def review_ratemaps(model_state_path: str):
    # Run tests on cpu
    device = torch.device("cpu")

    model = gridcell_models.DeepMindModel()
    model_state = torch.load(model_state_path)
    model.load_state_dict(model_state)
    model.cpu().eval()

    # Make a dataset from a single path that was
    # not used during trainig and validation
    paths = glob('data/encoded_pickles/*pickle')
    path = paths[-1]
    batch = make_test_batch(path)

    draw_ratemaps(model, device, batch)
