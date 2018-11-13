import numpy as np
import quantities as pq
import neo


def test_spatial_rate_map_1d():
    from exana.tracking.fields import spatial_rate_map_1d
    track_len = 10. * pq.m
    t = np.arange(10) * pq.s
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) * pq.m
    sptr = neo.SpikeTrain(times=np.linspace(0., 2, 10)*pq.s,
                          t_stop=10*pq.s)
    binsize = 5 * pq.m
    rate, bins = spatial_rate_map_1d(
        x, t, sptr,
        binsize=binsize,
        track_len=track_len,
        mask_unvisited=True,
        convolve=False,
        return_bins=True,
        smoothing=0.02)
    assert np.array_equal(rate, [1., 0])
