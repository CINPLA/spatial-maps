import numpy as np
import pytest


def test_calc_population_vector_correlation():
    from stats import pvcorr
    rmaps1 = np.array([
        [
            [1, 0],
            [0, 4]
        ],
        [
            [0, 0],
            [1, 2]
        ]])
    rmaps2 = np.array([
        [
            [2, 0],
            [0, 8]
        ],
        [
            [0, 0],
            [0.5, 0.5]
        ]])
    pv = pvcorr(rmaps1, rmaps2, minrate=1.)
    assert pv == 1
    
    
