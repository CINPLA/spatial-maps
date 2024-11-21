import pytest
import numpy as np


def test_border_score():
    from spatial_maps.bordercells import border_score
    from spatial_maps.tools import make_test_border_map
    from spatial_maps.fields import separate_fields_by_laplace
    box_size = [1., 1.]
    rate = 1.
    bin_size = [.01, .01]

    rate_map, pos_true, xbins, ybins = make_test_border_map(
        sigma=0.05, amplitude=rate, offset=0, box_size=box_size,
        bin_size=bin_size)

    labels = separate_fields_by_laplace(rate_map, threshold=0)
    bs = border_score(rate_map, labels)
    assert round(bs, 2) == .32
