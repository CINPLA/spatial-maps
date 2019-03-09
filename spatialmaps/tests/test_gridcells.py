import numpy as np
import pytest
from spatialmaps import SpatialMap
import quantities as pq
from tools import make_test_grid_rate_map, make_test_spike_map

def test_gridness():
    from spatialmaps.gridcells import gridness
    box_size = [1., 1.]
    rate = 5.
    bin_size = [.01, .01]

    rate_map, pos_fields, xbins, ybins = make_test_grid_rate_map(
        sigma=0.05, spacing=0.3, amplitude=rate, offset=0, box_size=box_size,
        bin_size=bin_size)

    g = gridness(rate_map)
    assert round(g, 2) == 1.3
