import json
import random
import datetime as dt
from glob import glob
from dataclasses import asdict, dataclass

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from gridcells.data import encoder as data_encoder
from gridcells.models import main as gridcell_models
from gridcells.data.dataset import EncodedLocationDataset
from gridcells.validation import views as validation_views
from gridcells.training.deepmind import epochs as training_epochs
from gridcells.training.deepmind import reviews as deepmind_reviews
from gridcells.training.base.rmsprop_tf import RMSprop as RMSprop_tf


@dataclass
class Config:
    batch_size: int = 10
    validation_batch_size: int = 500
    n_epochs: int = 301
    samples_per_epoch: int = 10_000
    validation_samples_per_epoch: int = 1000
    learning_rate: float = 1e-4

    use_dropout: bool = True
    weight_decay: float = 1e-5

    position_encoding_size: int = 256
    seed: int = 42
    # set seed to None if you want a random seed

    def markdown(self) -> str:
        d = asdict(self)

        # Tensorboard needs markdown without any indents :(
        text = f"""
### Experiment config

```
{json.dumps(d, indent=4)}
```
        """
        return text


def train():
    config = Config(
        batch_size=10,
        validation_batch_size=500,
        n_epochs=301,
        samples_per_epoch=10_000,
        position_encoding_size=256,
    )

    if config.seed is not None:
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    date_time = dt.datetime.now().strftime("%m%d_%H%M")
    if device.type == "cuda":
        run_name = "GPU_" + str(torch.cuda.current_device()) + "_DM_" + date_time
    else:
        run_name = "CPU_DM_" + date_time
    print("Current run:", run_name)

    # Display settings in tensorboard/text
    writer = SummaryWriter(f"tmp/tensorboard/{run_name}")
    writer.add_text("config", config.markdown(), 0)

    paths = glob("data/torch/*pt")
    encoder = data_encoder.DeepMindishEncoder(
        n_place_cells=config.position_encoding_size,
    )

    t_dataset = EncodedLocationDataset(paths[:30], encoder)
    v_dataset = EncodedLocationDataset(paths[30:32], encoder)
    test_batch = make_test_batch(paths[44], encoder, n_samples=2000)

    train_loader = DataLoader(
        t_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        v_dataset,
        batch_size=config.validation_batch_size,
        shuffle=False,
    )

    model = gridcell_models.DeepMindModel(
        weight_decay=config.weight_decay,
        use_dropout=config.use_dropout,
        position_encoding_size=config.position_encoding_size,
    )
    model = model.to(device)

    # Default value in tensorflow is different than default value in torch
    # it's also called *rho* rather than *alpha* (I think)
    alpha = 0.9
    momentum = 0.9
    eps = 1e-10
    optimizer = RMSprop_tf(
        params=model.parameters(),
        lr=config.learning_rate,
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
            samples_per_epoch=config.samples_per_epoch,
        )

        validation_loss = training_epochs.validation_epoch(
            model=model,
            data_loader=validation_loader,
            device=device,
            samples_per_epoch=config.validation_samples_per_epoch,
        )

        writer.add_scalar("training/loss", training_loss, epoch)
        writer.add_scalar("validation/accuracy", validation_loss, epoch)

        epoch_summary = f"Training: {training_loss:.2f}, validation: {validation_loss:.2f}"
        progress_bar.set_description(epoch_summary)

        # Detailed validation
        if epoch % 10 == 0:
            save_experiment(model, optimizer, config, run_name)
            ratemaps = deepmind_reviews.make_scored_ratemaps(model, device, test_batch)

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
            deepmind_reviews.review_path_integration_batch(
                model=model,
                batch=test_batch,
                device=device,
                writer=writer,
                epoch=epoch,
                n_place_cells=config.position_encoding_size,
            )

    save_experiment(model, optimizer, config, run_name)

    return model


def save_experiment(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Config,
    run_name: str,
):
    experiment_state = {
        "config": config,
        "optimizer": optimizer.state_dict(),
        "model": model.state_dict(),
    }
    torch.save(experiment_state, f"tmp/{run_name}.pt")


def make_test_batch(path: str, encoder, n_samples: int = 5000) -> dict:
    dataset = EncodedLocationDataset([path], encoder)
    loader = DataLoader(dataset, batch_size=n_samples, shuffle=True)

    batch = next(iter(loader))

    return batch
