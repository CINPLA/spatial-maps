import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve_fft


def _adjust_bin_size(box_size, bin_size=None, bin_count=None):
    if bin_size is None and box_size is None:
        raise ValueError("Bin size or box size must be set")

    if isinstance(bin_size, (float, int)):
        bin_size = np.array([bin_size, bin_size])

    if isinstance(bin_count, int):
        bin_count = np.array([bin_count, bin_count])

    if isinstance(box_size, (float, int)):
        box_size = np.array([box_size, box_size])

    if bin_size is None:
        bin_size = np.array([box_size[0] / bin_count[0],
                             box_size[1] / bin_count[1]])

    # round bin size of to closest requested bin size
    bin_size = np.array([box_size[0] / int(box_size[0] / bin_size[0]),
                         box_size[1] / int(box_size[1] / bin_size[1])])

    return box_size, bin_size


def _make_bins(box_size, bin_size, bin_count):
    box_size, bin_size = _adjust_bin_size(box_size, bin_size, bin_count)
    xbins = np.arange(0, box_size[0] + bin_size[0], bin_size[0])
    ybins = np.arange(0, box_size[1] + bin_size[1], bin_size[1])
    return xbins, ybins


def smooth_map(rate_map, bin_size, smoothing):
    std_dev_pixels = smoothing / bin_size
    kernel = Gaussian2DKernel(std_dev_pixels[0], std_dev_pixels[1])
    return convolve_fft(rate_map, kernel)


def _occupancy_map(x, y, t, xbins, ybins):
    t_ = np.append(t, t[-1] + np.median(np.diff(t)))
    time_in_bin = np.diff(t_)
    values, _, _ = np.histogram2d(
        y, x, bins=[xbins, ybins], weights=time_in_bin)
    return values


def _spike_map(x, y, t, spike_times, xbins, ybins):
    t_ = np.append(t, t[-1] + np.median(np.diff(t)))
    spikes_in_bin, _ = np.histogram(spike_times, t_)
    values, _, _ = np.histogram2d(
        y, x, bins=[xbins, ybins], weights=spikes_in_bin)
    return values


class SpatialMap:
    def __init__(self, x, y, t, spike_times, box_size, bin_size, bin_count=None):
        xbins, ybins = _make_bins(box_size, bin_size, bin_count)
        self.spike_pos = _spike_map(x, y, t, spike_times, xbins, ybins)
        self.time_pos = _occupancy_map(x, y, t, xbins, ybins)
        assert all(self.spike_pos[self.time_pos == 0] == 0)

        self.bin_size = bin_size
        self.box_size = box_size

    def spike_map(self, smoothing):
        if smoothing == 0.0:
            return self.spike_pos

        return smooth_map(self.spike_pos, self.bin_size, smoothing)

    def occupancy_map(self, smoothing):
        if smoothing == 0.0:
            return self.time_pos

        return smooth_map(self.time_pos, self.bin_size, smoothing)

    def rate_map(self, smoothing, mask_zero_occupancy=False):
        spike_map = self.spike_map(smoothing)
        occupancy_map = self.occupancy_map(smoothing)
        if mask_zero_occupancy:
            # to avoid infinity (x/0) we set zero occupancy to nan
            # this can be handy when analyzing low occupancy maps
            occupancy_map[self.time_pos == 0] = np.nan
        rate_map = spike_map / occupancy_map
        return rate_map
