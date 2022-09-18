import pickle
import numpy as np
from dataclasses import dataclass
from matplotlib import pyplot as plt

from gridcells.data.encoder import DeepMindPlaceEncoder


@dataclass
class TensorflowOutput:
    encoded_target_pos: np.array
    encoded_target_hd: np.array
    logits_pos: np.array
    target_pos: np.array
    logits_hd: np.array

    def __rich_repr__(self):
        yield "target_pos", self.target_pos.shape
        yield "logits_hd", self.logits_hd.shape
        yield "logits_pos", self.logits_pos.shape
        yield "encoded_target_hd", self.encoded_target_hd.shape
        yield "encoded_target_pos", self.encoded_target_pos.shape

    def __getitem__(self, idx: int) -> "TensorflowOutput":
        return TensorflowOutput(
            encoded_target_pos=self.encoded_target_pos[idx],
            encoded_target_hd=self.encoded_target_hd[idx],
            logits_pos=self.logits_pos[idx],
            target_pos=self.target_pos[idx],
            logits_hd=self.logits_hd[idx],
        )


def load_data(path: str) -> TensorflowOutput:
    """
        This has to be synchronized with the original deepmind repository

        Here's the piece of code I used to extract data from
        tensorflow objects into dicts:

        res = sess.run(
            {
                "encoded_target_pos": ensembles_targets[0],
                "encoded_target_hd": ensembles_targets[1],
                "logits_pos": ensembles_logits[0],
                "logits_hd": ensembles_logits[1],
                "target_pos": target_pos,
            }
        )

        This is the structure that is loaded here into the TensorflowOutput.
    """
    with open(path, "rb") as f:
        res = pickle.load(f, encoding='latin1')

    data = TensorflowOutput(**res)

    return data


def position_mse(data: TensorflowOutput):
    position_encoder = DeepMindPlaceEncoder()

    predicted_xy = position_encoder.decode(data.logits_pos)
    true_xy = position_encoder.decode(data.encoded_target_pos)

    # Error of quantization only
    quantization_mse = np.mean((true_xy - data.target_pos) ** 2)

    # Error of predicted encoded positions
    prediction_mse = np.mean((true_xy - predicted_xy) ** 2)

    print('Quantization MSE:', quantization_mse)
    print('Prediction MSE:', prediction_mse)


def position_outputs_plot(
    target_pos: np.array,
    logits_pos: np.array,
    encoded_target_pos: np.array,
):
    position_encoder = DeepMindPlaceEncoder()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    predicted_xy = position_encoder.decode(logits_pos)
    true_xy = position_encoder.decode(encoded_target_pos)

    ax.plot(predicted_xy[:, 0], predicted_xy[:, 1], label='predicted')
    ax.plot(true_xy[:, 0], true_xy[:, 1], label='encoded trajectory')
    ax.plot(target_pos[:, 0], target_pos[:, 1], label='true trajectory')

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    ax.legend()

    return fig
