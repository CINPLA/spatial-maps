import numpy as np
import pytest
import quantities as pq
from tools import make_test_grid_rate_map, make_test_border_map
from spatial_maps.fields import (
    separate_fields_by_laplace, find_peaks, calculate_field_centers,
    border_score)


def test_find_peaks():
    box_size = [1., 1.]
    rate = 5.
    bin_size = [.01, .01]

    rate_map, pos_fields, xbins, ybins = make_test_grid_rate_map(
        sigma=0.05, spacing=0.3, amplitude=rate, offset=0, box_size=box_size,
        bin_size=bin_size)
    peaks = find_peaks(rate_map)
    pos_peaks = np.array([xbins[peaks[:,0]], ybins[peaks[:,1]]]).T
    assert all(
        [np.isclose(p, pos_peaks, rtol=1e-3).prod(axis=1).any()
         for p in pos_fields])


def test_separate_fields_by_laplace():
    box_size = [1., 1.]
    rate = 1.
    bin_size = [.01, .01]

    rate_map, pos_true, xbins, ybins = make_test_grid_rate_map(
        sigma=0.05, spacing=0.3, amplitude=rate, offset=0, box_size=box_size,
        bin_size=bin_size)

    labels = separate_fields_by_laplace(rate_map, threshold=0)
    peaks = calculate_field_centers(rate_map, labels)
    bump_centers = np.array([xbins[peaks[:,0]], ybins[peaks[:,1]]])
    # The position of a 2D bin is defined to be its center
    for p in pos_true:
        assert np.isclose(p, pos_true).prod(axis=1).any()


def test_border_score():
    box_size = [1., 1.]
    rate = 1.
    bin_size = [.01, .01]

    rate_map, pos_true, xbins, ybins = make_test_border_map(
        sigma=0.05, amplitude=rate, offset=0, box_size=box_size,
        bin_size=bin_size)

    labels = separate_fields_by_laplace(rate_map, threshold=0)
    bs = border_score(rate_map, labels)
    assert round(bs, 2) == .32
