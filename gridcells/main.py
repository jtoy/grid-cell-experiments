import os
import pickle
from glob import glob

import scipy
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader

from gridcells.data import encoder as data_encoder
from gridcells.models import main as gridcell_models
from gridcells.training.base import main as base_training
from gridcells.validation import views as validation_views
from gridcells.training.deepmind import main as deepmind_training
from gridcells.training.ganguli import main as ganguli_training
from gridcells.data.dataset import SelfLocationDataset, EncodedLocationDataset


def review_decoding():
    paths = glob("data/torch/*pt")
    dataset = SelfLocationDataset(paths[:1])
    idx = np.random.randint(len(dataset))
    target_pos = dataset[idx]["target_pos"]
    position_encoder = data_encoder.DeepMindPlaceEncoder()
    validation_views.review_position_encoder(target_pos, position_encoder)


def lstm_pipeline_prototype():
    paths = glob("data/torch/*pt")
    encoder = data_encoder.DeepMindishEncoder()
    dataset = EncodedLocationDataset(paths, encoder)

    loader = DataLoader(dataset, batch_size=1047)
    batch = next(iter(loader))

    # In the original code the concatenation of initial conditions
    # happens within the model code ...
    encoded_inits = batch["encoded_inits"]
    encoded_pos = encoded_inits["position"].float()
    encoded_hd = encoded_inits["head_direction"].float()
    concat_init = torch.cat([encoded_hd, encoded_pos], axis=2).squeeze()

    # The sequence for the RNN is the list of agent velocities
    # at every timestep (all trajectories have 100 steps)
    ego_vel = batch["ego_vel"].float()

    model = gridcell_models.DeepMindModel()

    predicted_positions, predicted_hd, bottlenecks = model(concat_init, ego_vel)

    target_pos = batch["encoded_targets"]["position"].float()
    target_hd = batch["encoded_targets"]["head_direction"].float()

    pc_loss = F.cross_entropy(predicted_positions.view(-1, 256), target_pos.argmax(2).view(-1))
    hd_loss = F.cross_entropy(predicted_hd.view(-1, 12), target_hd.argmax(2).view(-1))

    loss = (pc_loss + hd_loss) / 2

    print("Place loss:", pc_loss.item())
    print("Head direction loss:", hd_loss.item())
    print("Total loss value:", loss.item())

    target_pos = batch["target_pos"]
    data_xy = target_pos.reshape(-1, target_pos.shape[-1])
    x = data_xy[:, 0]
    y = data_xy[:, 1]
    activations = bottlenecks.reshape(-1, 256)

    coord_range = ((-1.1, 1.1), (-1.1, 1.1))
    nbins = 20
    n_neurons = 256
    rate_maps = []
    for it in range(n_neurons):
        rate_map = scipy.stats.binned_statistic_2d(
            x.detach().numpy(),
            y.detach().numpy(),
            activations.detach().numpy()[:, it],
            bins=nbins,
            statistic="mean",
            range=coord_range,
        )[0]
        rate_maps.append(rate_map)


def cache_encoded_dataset():
    paths = glob("data/torch/*pt")
    batch_size = 10_000

    # TODO This could be a CLI argument
    n_place_cells = 32 ** 2
    encoder = data_encoder.DeepMindishEncoder(n_place_cells=n_place_cells)
    dataset = EncodedLocationDataset(paths, encoder)
    loader = DataLoader(dataset, batch_size=batch_size)

    savedir = f"data/encoded_pickles_{n_place_cells}"
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    for it, batch in tqdm(enumerate(loader), total=len(loader)):
        savepath = f"{savedir}/{it:03}.pickle"
        with open(savepath, "wb") as f:
            pickle.dump(batch, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--process", required=False, type=str)
    args = parser.parse_args()

    if args.process == "encode_dataset":
        cache_encoded_dataset()
    elif args.process == "baseline_train":
        base_training.train(n_epochs=1001)
    elif args.process == "ganguli_train":
        ganguli_training.train(n_epochs=1001)
    else:
        deepmind_training.train()
