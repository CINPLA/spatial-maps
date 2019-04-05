import numpy as np
import pytest
from spatial_maps import SpatialMap
import quantities as pq
from spatial_maps.tools import (
    make_test_grid_rate_map, make_test_spike_map, autocorrelation)
from spatial_maps.fields import find_peaks
from spatial_maps.gridcells import (
    gridness, spacing_and_orientation, separate_fields_by_distance)


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


def test_spacing_and_orientation_from_true_peaks():
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


def test_spacing_and_orientation_from_autocorr():
    box_size = np.array([1., 1.])
    rate = 5.
    bin_size = [.01, .01]
    spacing_true = 0.3
    orientation_true = .3

    rate_map, pos_fields, xbins, ybins = make_test_grid_rate_map(
        sigma=0.05, spacing=spacing_true, amplitude=rate, offset=0, box_size=box_size,
        bin_size=bin_size, orientation=orientation_true)
    autocorrelogram = autocorrelation(rate_map)
    peaks = find_peaks(autocorrelogram)
    real_peaks = peaks * bin_size
    autocorrelogram_box_size = box_size * autocorrelogram.shape[0] / rate_map.shape[0]
    spacing, orientation = spacing_and_orientation(real_peaks, autocorrelogram_box_size)
    assert round(spacing, 1) == spacing_true
    assert round(orientation, 1) == orientation_true


def test_separate_fields_by_distance():
    box_size = [1., 1.]
    rate = 1.
    bin_size = [.01, .01]

    rate_map, pos_true, xbins, ybins = make_test_grid_rate_map(
        sigma=0.05, spacing=0.3, amplitude=rate, offset=0, box_size=box_size,
        bin_size=bin_size)

    peaks, radius = separate_fields_by_distance(rate_map)
    bump_centers = np.array([xbins[peaks[:,0]], ybins[peaks[:,1]]])
    # The position of a 2D bin is defined to be its center
    for p in pos_true:
        assert np.isclose(p, pos_true).prod(axis=1).any()


def test_separate_fields_by_distance_2():
    Y, X = np.mgrid[0:100, 0:100]
    fx, fy = np.mgrid[5:95:20, 5:95:20]
    fields = np.array([fx.ravel(), fy.ravel()]).T

    rate_map = np.zeros((100, 100))

    for field in fields:
        dY = Y - field[0]
        dX = X - field[1]
        rate_map += np.exp(-1/2*(dY**2 + dX**2)/10)  # Gaussian-ish

    # should be removed by the algorithm because they are lower and close to existing fields
    noise_fields = [
        [60, 52],
        [45, 35]
    ]

    for field in noise_fields:
        dY = Y - field[0]
        dX = X - field[1]
        rate_map += 0.5 * np.exp(-1/2*(dY**2 + dX**2)/10)  # Gaussian-ish

    found_fields, radius = separate_fields_by_distance(rate_map)

    for field in found_fields:
        assert np.isclose(field, fields).prod(axis=1).any()
