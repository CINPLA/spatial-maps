import numpy as np
import pytest
import quantities as pq
from spatial_maps.tools import make_test_grid_rate_map, make_test_border_map
from spatial_maps.fields import (
    separate_fields_by_laplace, find_peaks, calculate_field_centers,
    in_field, distance_to_edge_function, map_pass_to_unit_circle)


def test_find_peaks():
    box_size = np.array([1., 1.])
    rate = 5.
    bin_size = [.01, .01]
    sigma=0.05
    spacing=0.3

    rate_map, pos_fields, xbins, ybins = make_test_grid_rate_map(
        sigma=sigma, spacing=spacing, amplitude=rate, offset=0, box_size=box_size,
        bin_size=bin_size, repeat=0)
    peaks = find_peaks(rate_map)
    pos_peaks = np.array([xbins[peaks[:,1]], ybins[peaks[:,0]]]).T
    print(pos_peaks)
    assert all(
        [np.isclose(p, pos_peaks, rtol=1e-3).prod(axis=1).any()
         for p in pos_fields])


def test_separate_fields_by_laplace():
    box_size = [1., 1.]
    rate = 1.
    bin_size = [.01, .01]
    sigma=0.05
    spacing=0.3

    rate_map, pos_true, xbins, ybins = make_test_grid_rate_map(
        sigma=sigma, spacing=spacing, amplitude=rate, offset=0.1, box_size=box_size,
        bin_size=bin_size, orientation=0.1)

    labels = separate_fields_by_laplace(rate_map, threshold=0)
    peaks = calculate_field_centers(rate_map, labels)
    bump_centers = np.array([xbins[peaks[:,0]], ybins[peaks[:,1]]])
    # The position of a 2D bin is defined to be its center
    for p in pos_true:
        assert np.isclose(p, pos_true).prod(axis=1).any()


def test_in_field():
    n_bins   = 10
    box_size = [1, 1]
    bin_size = box_size[0] / n_bins

    fields     = np.zeros((n_bins, n_bins))
    fields[:5] = 1
    fields[7]  = 2

    # pick out center of bins
    x = np.arange(bin_size/2, box_size[0], bin_size)
    y = box_size[0] / 2 * np.ones_like(x)
    true_value = [1, 1, 1, 1, 1, 0, 0, 2, 0, 0]
    assert np.all(in_field(x, y, fields, box_size) == true_value)
    # test edges
    x = np.array([0, 1])
    y = np.array([0.5, 0.5])
    fields[:] = 0
    fields[0] = 1
    fields[-1] = 2
    assert np.all(in_field(x, y, fields,box_size) == [1, 2])


def test_distance_to_edge_function():
    n_bins   = 10
    box_size = [1, 1]
    bin_size = box_size[0] / n_bins

    field    = np.zeros((n_bins, n_bins))
    field[2:8, 2:8] = 1
    d = distance_to_edge_function(
        0.5, 0.5, field, box_size, interpolation='linear')

    # assert edges 3/10 of the box size from center
    for a in [i * np.pi / 2 for i in range(4)]:
        assert np.isclose(0.3, d(a))

    # assert area within 5 % of expected result
    angles = np.linspace(0, 2 * np.pi, 10000)
    dist = d(angles)
    x = dist * np.cos(angles)
    y = dist * np.sin(angles)

    dx = np.gradient(x)
    dy = np.gradient(y)

    # Greens theorem
    area = 0.5 * np.sum(x * dy - y * dx)
    exact_area = np.sum(field) / np.size(field) * box_size[0]**2

    assert np.abs(area - exact_area) / exact_area < 0.05

def test_map_pass_to_unit_circle():

    dist_func = lambda theta : 1
    x_c, y_c = (0.5, 0.5)
    theta = np.linspace(np.pi, 2 * np.pi, 100)
    t = theta
    x = x_c + np.cos(theta)
    y = y_c + np.sin(theta)

    r, angle, _, _ = map_pass_to_unit_circle(x, y, t, x_c, y_c, dist_func=dist_func)

    assert np.all(np.isclose(angle, theta % (2 * np.pi)))
    assert np.all(np.isclose(1, r))
