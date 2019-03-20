import numpy as np
import pytest
import pdb

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
            [0, 0],
            [1, 2]
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
            [0, 0],
            [0.5, 0.5]
        ]])
    pv = pvcorr(rmaps1, rmaps2, min_rate=1.)
    err = pv-1
    assert err < 10e-5
    
    
def test_max_of_planes_in_cube():
    from spatial_maps.stats import _max_of_planes_in_cube as mpc
    
    c = np.arange(0, 18, 1.)
    c = c.reshape([2, 3, 3])

    c[0, 2, 2] = np.nan

    max_s0 = mpc(c)
    assert np.array_equal(max_s0, np.array([7, 17]))


def test_mask_in_both_if_any_is_nan():
    from spatial_maps.stats import _mask_in_both_if_any_is_nan as mask
    
    c0 = np.arange(0, 18, 1.)
    c0 = c0.reshape([2, 3, 3])
    c0[0, 2, 2] = np.nan

    c1 = np.arange(0, 18, 1.)
    c1 = c1.reshape([2, 3, 3])
    c1[1, 0, 0] = np.nan
    
    mask_target = np.ones((2, 3, 3), dtype=bool)
    mask_target[0, 2, 2] = False
    mask_target[1, 0, 0] = False

    assert np.array_equal(mask_target, mask(c0, c1))
    
    
