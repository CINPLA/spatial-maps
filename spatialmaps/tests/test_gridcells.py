import numpy as np
import pytest
from spatialmaps import SpatialMap
import quantities as pq
from tools import make_test_grid_rate_map, make_test_spike_map
from spatialmaps.gridcells import (
    gridness, spacing_and_orientation, separate_fields_from_distance)


def test_gridness():
    box_size = np.array([1., 1.])
    rate = 5.
    bin_size = [.01, .01]
    spacing_true = 0.3

    rate_map, pos_fields, xbins, ybins = make_test_grid_rate_map(
        sigma=0.05, spacing=spacing_true, amplitude=rate, offset=0, box_size=box_size,
        bin_size=bin_size)

    g = gridness(rate_map)
    assert round(g, 1) == 1.3


def test_spacing_and_orientation():
    box_size = np.array([1., 1.])
    rate = 5.
    bin_size = [.01, .01]
    spacing_true = 0.3

    rate_map, pos_fields, xbins, ybins = make_test_grid_rate_map(
        sigma=0.05, spacing=spacing_true, amplitude=rate, offset=0, box_size=box_size,
        bin_size=bin_size)

    spacing, orientation = spacing_and_orientation(pos_fields, box_size)
    assert spacing == spacing_true
    assert round(orientation * 180 / np.pi) == 30


def test_separate_fields_from_distance():
    box_size = [1., 1.]
    rate = 1.
    bin_size = [.01, .01]

    rate_map, pos_true, xbins, ybins = make_test_grid_rate_map(
        sigma=0.05, spacing=0.3, amplitude=rate, offset=0, box_size=box_size,
        bin_size=bin_size)

    peaks, radius = separate_fields_from_distance(rate_map)
    bump_centers = np.array([xbins[peaks[:,0]], ybins[peaks[:,1]]])
    # The position of a 2D bin is defined to be its center
    for p in pos_true:
        assert np.isclose(p, pos_true).prod(axis=1).any()
