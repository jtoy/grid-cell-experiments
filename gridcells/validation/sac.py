import math
import scipy
import numpy as np

from dataclasses import dataclass


def calculate_ratemap(xs: np.array, ys: np.array, activations: np.array):
    coord_range = ((-1.1, 1.1), (-1.1, 1.1))
    ratemap = scipy.stats.binned_statistic_2d(
        xs,
        ys,
        activations,
        bins=20,
        statistic='mean',
        range=coord_range,
    )[0]
    return ratemap


def calculate_sac(seq1):
    """Calculating spatial autocorrelogram."""
    seq2 = seq1

    def filter2(b, x):
        stencil = np.rot90(b, 2)
        return scipy.signal.convolve2d(x, stencil, mode='full')

    seq1 = np.nan_to_num(seq1)
    seq2 = np.nan_to_num(seq2)

    ones_seq1 = np.ones(seq1.shape)
    ones_seq1[np.isnan(seq1)] = 0
    ones_seq2 = np.ones(seq2.shape)
    ones_seq2[np.isnan(seq2)] = 0

    seq1[np.isnan(seq1)] = 0
    seq2[np.isnan(seq2)] = 0

    seq1_sq = np.square(seq1)
    seq2_sq = np.square(seq2)

    seq1_x_seq2 = filter2(seq1, seq2)
    sum_seq1 = filter2(seq1, ones_seq2)
    sum_seq2 = filter2(ones_seq1, seq2)
    sum_seq1_sq = filter2(seq1_sq, ones_seq2)
    sum_seq2_sq = filter2(ones_seq1, seq2_sq)
    n_bins = filter2(ones_seq1, ones_seq2)
    n_bins_sq = np.square(n_bins)

    std_seq1 = np.power(
        np.subtract(
            np.divide(sum_seq1_sq, n_bins),
            (np.divide(np.square(sum_seq1), n_bins_sq))), 0.5)
    std_seq2 = np.power(
        np.subtract(
            np.divide(sum_seq2_sq, n_bins),
            (np.divide(np.square(sum_seq2), n_bins_sq))), 0.5)
    covar = np.subtract(
        np.divide(seq1_x_seq2, n_bins),
        np.divide(np.multiply(sum_seq1, sum_seq2), n_bins_sq))
    x_coef = np.divide(covar, np.multiply(std_seq1, std_seq2))
    x_coef = np.real(x_coef)
    x_coef = np.nan_to_num(x_coef)
    return x_coef


def _rotate_sac(sac: np.array, angle: int) -> np.array:
    out = scipy.ndimage.interpolation.rotate(sac, angle, reshape=False)
    return out


def circle_mask(size, radius, in_val=1.0, out_val=0.0):
    """Calculating the grid scores with different radius."""
    sz = [math.floor(size[0] / 2), math.floor(size[1] / 2)]
    x = np.linspace(-sz[0], sz[1], size[1])
    x = np.expand_dims(x, 0)
    x = x.repeat(size[0], 0)
    y = np.linspace(-sz[0], sz[1], size[1])
    y = np.expand_dims(y, 1)
    y = y.repeat(size[1], 1)
    z = np.sqrt(x**2 + y**2)
    z = np.less_equal(z, radius)
    vfunc = np.vectorize(lambda b: b and in_val or out_val)
    return vfunc(z)


@dataclass
class Ratemap:
    ratemap: np.array
    s60: float
    s90: float

    def __rich_repr__(self):
        yield "Ratemap Gridscores"
        yield "ratemap", self.ratemap.shape
        yield "s60", self.s60
        yield "s90", self.s90


class GridScorer:
    def __init__(self, nbins: int = 20):
        self.nbins = nbins
        self.n_points = [nbins * 2 - 1, nbins * 2 - 1]
        self.angles = [30, 45, 60, 90, 120, 135, 150]

        # Uses a sequence of growing rings to measure grid score
        starts = [0.2] * 10
        ends = np.linspace(0.4, 1.0, num=10)
        mask_parameters = zip(starts, ends.tolist())

        self.ring_masks = [
            self._get_ring_mask(mask_min, mask_max)
            for mask_min, mask_max in mask_parameters
        ]

    def _get_ring_mask(self, mask_min, mask_max):
        mask = circle_mask(self.n_points, mask_max * self.nbins) * (
            1 - circle_mask(self.n_points, mask_min * self.nbins)
        )
        return mask

    def get_sac_angle_correlations(self, sac: np.array, mask: np.array) -> dict:
        """Calculate Pearson correlations of area inside mask at corr_angles."""
        masked_sac = sac * mask
        ring_area = np.sum(mask)

        # Calculate dc on the ring area
        masked_sac_mean = np.sum(masked_sac) / ring_area

        # Center the sac values inside the ring
        masked_sac_centered = (masked_sac - masked_sac_mean) * mask
        variance = np.sum(masked_sac_centered**2) / ring_area + 1e-5
        corrs = dict()

        # for angle, rotated_sac in zip(self._corr_angles, rotated_sacs):
        for angle in self.angles:
            rotated_sac = _rotate_sac(sac, angle)
            masked_rotated_sac = (rotated_sac - masked_sac_mean) * mask
            cross_prod = np.sum(masked_sac_centered * masked_rotated_sac) / ring_area
            corrs[angle] = cross_prod / variance
        return corrs

    def grid_score_60(self, corr: dict) -> float:
        s60 = (corr[60] + corr[120]) / 2 - (corr[30] + corr[90] + corr[150]) / 3
        return s60

    def grid_score_90(self, corr):
        return corr[90] - (corr[45] + corr[135]) / 2

    def get_grid_score(self, ratemap: np.array) -> Ratemap:
        sac = calculate_sac(ratemap)
        scores = []
        for mask in self.ring_masks:
            masked_correlations = self.get_sac_angle_correlations(sac, mask)
            s60 = self.grid_score_60(masked_correlations)
            s90 = self.grid_score_90(masked_correlations)
            score = {
                's60': s60,
                's90': s90,
            }
            scores.append(score)

        s60 = max([s['s60'] for s in scores])
        s90 = max([s['s90'] for s in scores])
        rated_map = Ratemap(
            ratemap=ratemap,
            s60=s60,
            s90=s90,
        )
        return rated_map
