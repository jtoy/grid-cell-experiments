import numpy as np
from scipy.special import softmax, logsumexp

from gridcells.data.structures import Trajectory

SEED = 8341


class DeepMindishEncoder:
    def __init__(self, n_place_cells: int = 1032, n_head_cells: int = 12):
        self.place_encoder = DeepMindPlaceEncoder(n_cells=n_place_cells)
        self.head_direction_encoder = DeepMindHeadEncoder(n_cells=n_head_cells)

    def encode(self, trajectory: Trajectory) -> dict:
        encoded_initial_pos = self.place_encoder.encode(trajectory.init_pos[np.newaxis, :])
        encoded_initial_hd = self.head_direction_encoder.encode(trajectory.init_hd)
        encoded_target_pos = self.place_encoder.encode(trajectory.target_pos)
        encoded_target_hd = self.head_direction_encoder.encode(trajectory.target_hd)

        record = {
            "encoded_initial_pos": encoded_initial_pos,
            "encoded_initial_hd": encoded_initial_hd,
            "encoded_target_pos": encoded_target_pos,
            "encoded_target_hd": encoded_target_hd,
        }
        return record


class DeepMindHeadEncoder:
    def __init__(self, n_cells: int = 12):
        self.n_cells = n_cells

        concentration = 20

        rs = np.random.RandomState(SEED)
        self.means = rs.uniform(-np.pi, np.pi, (n_cells))
        self.kappa = np.ones_like(self.means) * concentration

    def encode(self, head_direction: np.array) -> np.array:
        logp = self.kappa * np.cos(head_direction - self.means[np.newaxis, :])
        log_posteriors = logp - logsumexp(logp, axis=1, keepdims=True)

        probs = softmax(log_posteriors, axis=-1)
        return probs.astype(np.float32)

    def decode(self, x: np.array) -> np.array:
        idxs = x[:, :].argmax(-1)
        recreated = np.array([self.means[idx] for idx in idxs])
        return recreated


class DeepMindPlaceEncoder:
    def __init__(self, n_cells: int = 256):
        self.n_cells = n_cells

        # Hard coded settings used in the original repo
        pos_min = -1.1
        pos_max = 1.1
        stdev = 0.01

        rs = np.random.RandomState(SEED)
        self.means = rs.uniform(pos_min, pos_max, size=(n_cells, 2))
        self.variances = np.ones_like(self.means) * stdev ** 2

    def encode(self, trajs: np.array):
        diff = trajs[:, np.newaxis, :] - self.means[np.newaxis, ...]
        normalized_diff = (diff ** 2) / self.variances

        logp = -0.5 * np.sum(normalized_diff, axis=-1)

        # TODO Make sure to do logsumexp on the "means" axis
        log_posteriors = logp - logsumexp(logp, axis=1, keepdims=True)
        probs = softmax(log_posteriors, axis=-1)

        return probs.astype(np.float32)

    def decode(self, x: np.array) -> np.array:
        idxs = x[:, :].argmax(-1)
        recreated = np.array([self.means[idx] for idx in idxs])
        return recreated
