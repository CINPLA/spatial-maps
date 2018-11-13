import numpy as np
import pytest


def test_cut_to_same_len():
    from exana.tracking.tools import _cut_to_same_len
    t = np.arange(12)
    x = np.arange(11)
    y = np.arange(11, 24)
    t_, x_, y_ = _cut_to_same_len(t, x, y)
    assert np.array_equal(t_, t[:-1])
    assert np.array_equal(x_, x)
    assert np.array_equal(y_, y[:-2])


def test_remove_eqal_times():
    from exana.tracking.tools import remove_eqal_times
    t = np.arange(20)
    t[4] = t[5]
    t[6] = t[5]
    t[8] = t[9]
    t[10] = t[9]
    x = np.arange(11)
    y = np.arange(11, 24)
    t_, (x_, y_) = remove_eqal_times(t, x, y)
    t = np.delete(t, [5, 6, 9, 10])
    x = np.delete(x, [5, 6, 9, 10])
    y = np.delete(y, [5, 6, 9, 10])
    assert np.array_equal(t_, t)
    assert np.array_equal(x_, x)
    assert np.array_equal(y_, y)


def test_remove_smaller_times():
    from exana.tracking.tools import remove_smaller_times
    t = np.arange(11)
    t[4] = t[2]
    t[6] = t[4]
    t[8] = t[2]
    t[10] = t[3]
    x = np.arange(11)
    y = np.arange(11, 22)
    t_, (x_, y_) = remove_smaller_times(t, x, y)
    t = np.delete(t, [4, 6, 8, 10])
    x = np.delete(x, [4, 6, 8, 10])
    y = np.delete(y, [4, 6, 8, 10])
    assert np.array_equal(t_, t)
    assert np.array_equal(x_, x)
    assert np.array_equal(y_, y)


def test_remove_smaller_times1():
    from exana.tracking.tools import remove_smaller_times
    t = [1,2,3,2,4]
    x = [1,2,3,2,4]
    y = [5,6,7,8,9]
    t, (x, y) = remove_smaller_times(t, x, y)
    t_ = [1,2,3,4]
    x_ = [1,2,3,4]
    y_ = [5,6,7,9]
    assert np.array_equal(t_, t)
    assert np.array_equal(x_, x)
    assert np.array_equal(y_, y)


def test_monotonously_increasing():
    from exana.tracking.tools import monotonously_increasing
    t = np.arange(12)
    assert monotonously_increasing(t)
    t[4] = t[5]
    assert not monotonously_increasing(t)


def test_rm_nans():
    """
    Test of rm_nans(x,y,t)
    """
    import quantities as pq
    from exana.tracking.tools import rm_nans

    x = np.arange(0., 10.) * 0.1 * pq.m
    y = np.arange(0., 10.) * 1.0 * pq.m
    t = np.arange(0., 10.) * 10. * pq.m

    x[[0, 3, 4]] = np.nan * pq.m
    y[[3, 5]] = np.nan * pq.m
    t[[1, 2, 3, 4]] = np.nan * pq.m

    x_e = np.arange(6., 10.) * 0.1 * pq.m
    y_e = np.arange(6., 10.) * 1.0 * pq.m
    t_e = np.arange(6., 10.) * 10. * pq.m

    np.testing.assert_equal((x_e, y_e, t_e), rm_nans(x, y, t))


def test_spatial_rate_map_rate_shape():
    import quantities as pq
    from exana.tracking.fields import spatial_rate_map
    import neo
    N = 20
    binsize = 0.2 * pq.m
    box_xlen = 1.0 * pq.m
    box_ylen = 1.0 * pq.m
    x = np.linspace(0., box_xlen.magnitude, N) * pq.m
    y = np.linspace(0., box_ylen.magnitude, N) * pq.m
    t = np.linspace(0, 10., N) * pq.s
    sptr = neo.SpikeTrain(times=np.linspace(0, 10., N*2) * pq.s,
                          t_stop=max(t))
    ratemap, xbins, ybins = spatial_rate_map(x, y, t, sptr, binsize=binsize,
                                             box_xlen=box_xlen,
                                             box_ylen=box_ylen,
                                             mask_unvisited=False,
                                             convolve=False,
                                             return_bins=True)
    assert all(np.diff(np.diag(ratemap)) < 1e-10)
    assert ratemap.shape == (int(box_xlen/binsize), int(box_ylen/binsize))
    ratemap1 = spatial_rate_map(x, y, t, sptr, binsize=binsize,
                                box_xlen=1.0*pq.m, box_ylen=1.0*pq.m,
                                mask_unvisited=False, convolve=False,
                                return_bins=False)
    assert all(np.diag(ratemap - ratemap1) < 1e-10)


def test_spatial_rate_map_rate_convolve():
    import quantities as pq
    from exana.tracking.fields import spatial_rate_map
    import neo
    N = 20
    binsize = 0.2 * pq.m
    box_xlen = 1.0 * pq.m
    box_ylen = 1.0 * pq.m
    x = np.linspace(0., box_xlen.magnitude, N) * pq.m
    y = np.linspace(0., box_ylen.magnitude, N) * pq.m
    t = np.linspace(0, 10., N) * pq.s
    sptr = neo.SpikeTrain(times=np.linspace(0, 10., N*2) * pq.s,
                          t_stop=max(t))
    ratemap2 = spatial_rate_map(x, y, t, sptr, binsize=binsize,
                                box_xlen=1.0*pq.m, box_ylen=1.0*pq.m,
                                mask_unvisited=False, convolve=True,
                                return_bins=False, smoothing=0.02)
    assert all(np.diff(np.diag(ratemap2)) < 0.02)


def test_spatial_rate_map_binsize_error():
    import quantities as pq
    from exana.tracking.fields import spatial_rate_map
    import neo
    # raise error if box length not multiple of binsize
    with pytest.raises(ValueError):
        N = 20
        binsize = 0.23 * pq.m
        box_xlen = 1.0 * pq.m
        box_ylen = 1.0 * pq.m
        x = np.linspace(0., box_xlen.magnitude, N) * pq.m
        y = np.linspace(0., box_ylen.magnitude, N) * pq.m
        t = np.linspace(0, 10., N) * pq.s
        sptr = neo.SpikeTrain(times=np.linspace(0, 10., N*2) * pq.s,
                              t_stop=max(t))
        ratemap, xbins, ybins = spatial_rate_map(x, y, t, sptr, binsize=binsize,
                                                 box_xlen=box_xlen,
                                                 box_ylen=box_ylen,
                                                 mask_unvisited=False,
                                                 convolve=False,
                                                 return_bins=True)


def test_spatial_rate_map_len_error():
    import quantities as pq
    from exana.tracking.fields import spatial_rate_map
    import neo
    # raise error if len(t) != len(x)
    with pytest.raises(ValueError):
        N = 20
        binsize = 0.2 * pq.m
        box_xlen = 1.0 * pq.m
        box_ylen = 1.0 * pq.m
        x = np.linspace(0., box_xlen.magnitude, N) * pq.m
        y = np.linspace(0., box_ylen.magnitude, N) * pq.m
        t = np.linspace(0, 10., N+1) * pq.s
        sptr = neo.SpikeTrain(times=np.linspace(0, 10., N*2) * pq.s,
                              t_stop=max(t))
        ratemap, xbins, ybins = spatial_rate_map(x, y, t, sptr, binsize=binsize,
                                                 box_xlen=box_xlen,
                                                 box_ylen=box_ylen,
                                                 mask_unvisited=False,
                                                 convolve=False,
                                                 return_bins=True)


def test_spatial_rate_map_size_error():
    import quantities as pq
    from exana.tracking.fields import spatial_rate_map
    import neo
    # raise error if box length smaller than path
    with pytest.raises(ValueError):
        N = 20
        binsize = 0.2 * pq.m
        box_xlen = 1.0 * pq.m
        box_ylen = 1.0 * pq.m
        x = np.linspace(0., box_xlen.magnitude+1, N) * pq.m
        y = np.linspace(0., box_ylen.magnitude, N) * pq.m
        t = np.linspace(0, 10., N) * pq.s
        sptr = neo.SpikeTrain(times=np.linspace(0, 10., N*2) * pq.s,
                              t_stop=max(t))
        ratemap, xbins, ybins = spatial_rate_map(x, y, t, sptr, binsize=binsize,
                                                 box_xlen=box_xlen,
                                                 box_ylen=box_ylen,
                                                 mask_unvisited=False,
                                                 convolve=False,
                                                 return_bins=True)


def test_occupancy_map():
    import quantities as pq
    from exana.tracking.fields import occupancy_map
    N = 3
    binsize = 0.5 * pq.m
    box_xlen = 1.5 * pq.m
    box_ylen = 1.5 * pq.m
    x = np.linspace(0+binsize.magnitude/2.,
                    box_xlen.magnitude-binsize.magnitude/2, N) * pq.m
    y = np.linspace(0+binsize.magnitude/2.,
                    box_ylen.magnitude-binsize.magnitude/2, N) * pq.m
    t = np.linspace(0, 10., N) * pq.s

    occmap, xbins, ybins = occupancy_map(x, y, t,
                                         binsize=binsize,
                                         box_xlen=box_xlen,
                                         box_ylen=box_ylen,
                                         convolve=False,
                                         return_bins=True)
    occmap_expected = np.array([[5, 0, 0],
                                [0, 5, 0],
                                [0, 0, 5]])
    assert np.array_equal(occmap, occmap_expected)


def test_nvisits_map():
    # run a testcase with a small trace along a 4 bin box
    import quantities as pq
    from exana.tracking.fields import nvisits_map
    N = 8
    binsize = 0.5 * pq.m
    box_xlen = 1. * pq.m
    box_ylen = 1. * pq.m
    x = np.array([0.25, 0.75, 0.75, 0.25, 0.25, 0.25, 0.25, 0.75]) * pq.m
    y = np.array([0.25, 0.25, 0.75, 0.75, 0.25, 0.25, 0.25, 0.75]) * pq.m
    t = np.linspace(0, 10., N) * pq.s
    nvisits_map, xbins, ybins = nvisits_map(x, y, t,
                                            binsize=binsize,
                                            box_xlen=box_xlen,
                                            box_ylen=box_ylen,
                                            return_bins=True)
    nvisits_map_expected = np.array([[2, 1],
                                     [1, 2]])
    assert np.array_equal(nvisits_map, nvisits_map_expected)
