import numpy as np
import pytest
from spatial_maps.maps import SpatialMap
import quantities as pq
from spatial_maps.tools import make_test_grid_rate_map, make_test_spike_map


def test_rate_map():
    box_size = [1., 1.]
    rate = 5.
    bin_size = [.01, .01]
    n_step=10**4
    step_size=.1
    sigma=0.1
    spacing=0.3
    smoothing = .03

    rate_map_true, pos_fields, xbins, ybins = make_test_grid_rate_map(
        sigma=sigma, spacing=spacing, amplitude=rate, box_size=box_size,
        bin_size=bin_size)

    x, y, t, spikes = make_test_spike_map(
        pos_fields=pos_fields, box_size=box_size, rate=rate,
        n_step=n_step, step_size=step_size, sigma=sigma)
    smap = SpatialMap(
        x, y, t, spikes, box_size=box_size, bin_size=bin_size)
    rate_map = smap.rate_map(smoothing)

    diff = rate_map_true - rate_map
    X, Y = np.meshgrid(xbins, ybins)

    samples = []
    for p in pos_fields:
        mask = np.sqrt((X - p[0])**2 + (Y - p[1])**2) < .1
        samples.append(diff[mask])
    peak_diff = np.abs(np.mean([s.min() for s in samples if s.size > 0]))
    assert peak_diff < 0.5


def test_spatial_rate_map_diag():
    N = 10
    bin_size = 1
    box_size = 1.0
    x = np.linspace(0., box_size, N)
    y = np.linspace(0., box_size, N)
    t = np.linspace(0.1, 10.1, N)
    spike_times = np.arange(0.1, 10.1, .5)
    map = SpatialMap(
        x, y, t, spike_times, box_size=box_size, bin_size=bin_size)
    ratemap = map.rate_map(0)
    print(ratemap)
    assert all(np.diff(np.diag(ratemap)) < 1e-10)
    assert ratemap.shape == (int(box_size / bin_size), int(box_size / bin_size))


def test_occupancy_map_diag():
    N = 3
    bin_size = .5
    box_size = 1.5
    x = np.linspace(0., box_size, N)
    y = np.linspace(0., box_size, N)
    t = np.linspace(0, 10., N)

    map = SpatialMap(
        x, y, t, [], box_size=box_size, bin_size=bin_size)
    occmap_expected = np.array([[5, 0, 0],
                                [0, 5, 0],
                                [0, 0, 5]])
    occmap = map.occupancy_map(0)
    assert np.array_equal(occmap, occmap_expected)
