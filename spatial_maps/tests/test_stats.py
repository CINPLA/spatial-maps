import numpy as np
import pytest


def test_calc_population_vector_correlation():
    from spatial_maps.stats import population_vector_correlation as pvcorr
    rmaps1 = np.array([
        [
            [1, 0.1],
            [0.1, 4]
        ],
        [
            [6, 0.1],
            [0.1, 2]
        ],
        [
            [2, 0.1],
            [0.1, 3]
        ]])
    rmaps2 = np.array([
        [
            [2, 0.2],
            [0.2, 8]
        ],
        [
            [12, 0.2],
            [0.2, 4]
        ],
        [
            [4, 0.2],
            [0.2, 6]
        ]])
    rmaps2 += 10e-5
    pv = pvcorr(rmaps1, rmaps2)
    err = pv-1
    assert err < 10e-5
