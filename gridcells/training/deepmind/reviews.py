from glob import glob

import tqdm
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from gridcells.validation import sac as SAC
from gridcells.data import encoder as data_encoder
from gridcells.models import main as gridcell_models
from gridcells.data.dataset import EncodedLocationDataset
from gridcells.validation import views as validation_views


def review_path_integration_batch(
    model: nn.Module,
    batch: dict,
    device: str,
    writer,
    epoch: int,
    n_place_cells: int,
):
    # Draw this many charts
    n_samples = 10
    ego_vel = batch["ego_vel"].to(device)
    encoded_pos = batch["encoded_initial_pos"].to(device)
    encoded_hd = batch["encoded_initial_hd"].to(device)
    concat_init = torch.cat([encoded_hd, encoded_pos], axis=2).squeeze()
    # And for drawing charts
    init_pos = batch["init_pos"].detach().numpy()
    target_hd = batch["target_hd"].numpy()
    target_pos = batch["target_pos"].numpy()

    predicted_positions, predicted_hd, bottlenecks = model(concat_init, ego_vel)
    predicted_positions = predicted_positions.cpu().detach().numpy()
    predicted_hd = predicted_hd.cpu().detach().numpy()
    hd_encoder = data_encoder.DeepMindHeadEncoder()
    position_encoder = data_encoder.DeepMindPlaceEncoder(n_place_cells)

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
        writer.add_figure(f"path_integration/{it}", fig, epoch)


def make_scored_ratemaps(model: nn.Module, device: torch.device, batch: dict) -> list[SAC.Ratemap]:
    ego_vel = batch["ego_vel"].to(device)
    encoded_pos = batch["encoded_initial_pos"].to(device)
    encoded_hd = batch["encoded_initial_hd"].to(device)
    concat_init = torch.cat([encoded_hd, encoded_pos], axis=2).squeeze()
    target_pos = batch["target_pos"].numpy()

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


def review_path_integration(experiment_state_path: str):
    # Run tests on cpu
    device = torch.device("cpu")

    # Draw this many charts
    n_samples = 10

    model = gridcell_models.DeepMindModel()
    experiment_state = torch.load(experiment_state_path)
    model.load_state_dict(experiment_state["model"])
    model.cpu().eval()

    config = experiment_state["config"]
    encoder = data_encoder.DeepMindishEncoder(
        n_place_cells=config.position_encoding_size,
    )

    # Make a dataset from a single path that was
    # not used during trainig and validation
    paths = glob("data/torch/*pt")
    path = paths[-1]
    dataset = EncodedLocationDataset([path], encoder)
    loader = DataLoader(dataset, batch_size=n_samples, shuffle=True)

    position_encoder = encoder.place_encoder
    hd_encoder = encoder.head_direction_encoder

    # Unpack data for model input
    batch = next(iter(loader))
    ego_vel = batch["ego_vel"].to(device)
    encoded_pos = batch["encoded_initial_pos"].to(device)
    encoded_hd = batch["encoded_initial_hd"].to(device)
    concat_init = torch.cat([encoded_hd, encoded_pos], axis=2).squeeze()
    # And for drawing charts
    init_pos = batch["init_pos"].detach().numpy()
    target_hd = batch["target_hd"].numpy()
    target_pos = batch["target_pos"].numpy()

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
        savepath = f"tmp/dp-review-{it:02}.png"
        fig.savefig(savepath)
    return savepath


def review_head_angle_activations(experiment_state_path: str):
    device = torch.device("cpu")
    experiment_state = torch.load(experiment_state_path)
    config = experiment_state["config"]

    model = gridcell_models.DeepMindModel(
        weight_decay=config.weight_decay,
        use_dropout=config.use_dropout,
        bottleneck_size=config.bottleneck_size,
        position_encoding_size=config.position_encoding_size,
    )
    model.load_state_dict(experiment_state["model"])
    model.cpu().eval()

    encoder = data_encoder.DeepMindishEncoder(
        n_place_cells=config.position_encoding_size,
    )

    # Make a dataset from a single path that was
    # not used during trainig and validation
    paths = glob("data/torch/*pt")
    path = paths[-1]
    dataset = EncodedLocationDataset([path], encoder)
    loader = DataLoader(dataset, batch_size=10000, shuffle=True)

    # Unpack data for model input
    batch = next(iter(loader))
    ego_vel = batch["ego_vel"].to(device)
    encoded_pos = batch["encoded_initial_pos"].to(device)
    encoded_hd = batch["encoded_initial_hd"].to(device)
    concat_init = torch.cat([encoded_hd, encoded_pos], axis=2).squeeze()
    target_hd = batch["target_hd"].numpy()
    # target_pos = batch["target_pos"].numpy()

    predicted_positions, predicted_hd, bottlenecks = model(concat_init, ego_vel)

    bottleneck_size = config.bottleneck_size
    activations = bottlenecks.reshape(-1, bottleneck_size).detach().cpu().numpy()
    hd = target_hd.reshape(-1)

    scores = []
    for it in range(-100, 100):
        jt = it + 1
        ids = (hd >= it * np.pi / 100) & (hd < jt * np.pi / 100)
        angled_activations = activations[ids]

        # "cells" are in the dim=1, so this is per-cell
        res = angled_activations.mean(0)
        scores.append(res)
    scores = np.array(scores)

    # Find the least noisy curves
    diffs = np.diff(scores, axis=0)
    noise_score = np.std(diffs, axis=0)

    n_samples = 10
    x = np.linspace(-np.pi, np.pi, 200)
    fig, axes = plt.subplots(n_samples, 1, figsize=(8, 12))
    for it in range(n_samples):
        ax = axes[it]
        idx = noise_score.argsort()[it]
        ax.plot(x, scores[:, idx], label=f"Cell: {idx}")
        ax.legend()

    ax.set_xlabel("Head Angle [rad]")
    fig.suptitle("Cell Activation x Head Direction")

    return fig


def head_rotation_animation(
    activations: np.array,
    target_pos: np.array,
    hd: np.array,
):
    scores = []
    for it in range(-100, 100):
        jt = it + 1
        ids = (hd >= it * np.pi / 100) & (hd < jt * np.pi / 100)
        angled_activations = activations[ids]
        res = angled_activations.mean(0)
        scores.append(res)
    scores = np.array(scores)

    data_xy = target_pos.reshape(-1, target_pos.shape[-1])
    x = data_xy[:, 0]
    y = data_xy[:, 1]

    # Find the least noisy curves
    diffs = np.diff(scores, axis=0)
    noise_score = np.std(diffs, axis=0)
    cell_ids = noise_score.argsort()[:64]

    ratemap_grids = []
    for it in range(-128, 128):
        jt = it + 1
        ids = (hd >= it * np.pi / 128) & (hd < jt * np.pi / 128)
        x_part = x[ids]
        y_part = y[ids]
        act_part = activations[ids]
        ratemaps = [SAC.calculate_ratemap(x_part, y_part, act_part[:, cid]) for cid in cell_ids]
        grid = np.vstack([np.hstack(ratemaps[8 * it : 8 * (it + 1)]) for it in range(8)])
        grid[np.isnan(grid)] = 0
        ratemap_grids.append(grid)

    ratemap_grids = np.stack(ratemap_grids)
    ratemap_grids -= ratemap_grids.min()
    ratemap_grids /= ratemap_grids.max()
    ratemap_grids *= 255

    for it in tqdm(range(256)):
        savepath = f"tmp/to_gif/frame_{it:04}.png"
        img = Image.fromarray(ratemap_grids[it])

        img.convert("RGB").save(savepath)

    # https://gif.ski/
    # gifski -o tmp/tmp.gif tmp/to_gif/frame_*.png

    return ratemap_grids
