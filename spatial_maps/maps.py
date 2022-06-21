import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve_fft
from copy import copy


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
        bin_size = np.array([box_size[0] / bin_count[0], box_size[1] / bin_count[1]])

    # round bin size of to closest requested bin size
    bin_size = np.array(
        [
            box_size[0] / int(box_size[0] / bin_size[0]),
            box_size[1] / int(box_size[1] / bin_size[1]),
        ]
    )

    return box_size, bin_size


def _make_bins(box_size, bin_size):
    xbins = np.arange(0, box_size[0] + bin_size[0], bin_size[0])
    ybins = np.arange(0, box_size[1] + bin_size[1], bin_size[1])
    return xbins, ybins


def smooth_map(rate_map, bin_size, smoothing, **kwargs):
    std_dev_pixels = smoothing / bin_size
    kernel = Gaussian2DKernel(std_dev_pixels[0], std_dev_pixels[1])
    return convolve_fft(rate_map, kernel, **kwargs)


def _occupancy_map(x, y, t, xbins, ybins):
    t_ = np.append(t, t[-1] + np.median(np.diff(t)))
    time_in_bin = np.diff(t_)
    values, _, _ = np.histogram2d(x, y, bins=[xbins, ybins], weights=time_in_bin)
    return values


def _spike_map(x, y, t, spike_times, xbins, ybins):
    t_ = np.append(t, t[-1] + np.median(np.diff(t)))
    spikes_in_bin, _ = np.histogram(spike_times, t_)
    values, _, _ = np.histogram2d(x, y, bins=[xbins, ybins], weights=spikes_in_bin)
    return values


def interpolate_nan_2D(array, method="nearest"):
    from scipy import interpolate

    x = np.arange(0, array.shape[1])
    y = np.arange(0, array.shape[0])
    # mask invalid values
    array = np.ma.masked_invalid(array)
    xx, yy = np.meshgrid(x, y)
    # get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]

    return interpolate.griddata(
        (x1, y1), newarr.ravel(), (xx, yy), method=method, fill_value=0
    )


class SpatialMap:
    def __init__(
        self, smoothing=0.05, box_size=[1.0, 1.0], bin_size=0.02, bin_count=None
    ):
        """
        Parameters
        ----------
        smoothing : float
            Smoothing of spike_map and occupancy_map before division
        box_size : Sequence-like
            Size of box in x and y direction
        bin_size : float
            Resolution of spatial maps
        """
        box_size, bin_size = _adjust_bin_size(box_size, bin_size, bin_count)
        xbins, ybins = _make_bins(box_size, bin_size)

        self.smoothing = smoothing
        self.bin_size = bin_size
        self.box_size = box_size
        self.xbins = xbins
        self.ybins = ybins

    def spike_map(self, x, y, t, spike_times, mask_zero_occupancy=True, **kwargs):
        spmap = _spike_map(x, y, t, spike_times, self.xbins, self.ybins)
        spmap = (
            smooth_map(spmap, self.bin_size, self.smoothing, **kwargs)
            if self.smoothing
            else spmap
        )
        if mask_zero_occupancy:
            spmap[_occupancy_map(x, y, t, self.xbins, self.ybins) == 0] = np.nan
        return spmap

    def occupancy_map(self, x, y, t, mask_zero_occupancy=True, **kwargs):
        ocmap = _occupancy_map(x, y, t, self.xbins, self.ybins)
        ocmap_copy = copy(ocmap)  # to mask zero occupancy after smoothing
        ocmap = (
            smooth_map(ocmap, self.bin_size, self.smoothing, **kwargs)
            if self.smoothing
            else ocmap
        )
        if mask_zero_occupancy:
            ocmap[ocmap_copy == 0] = np.nan
        return ocmap

    def rate_map(
        self,
        x,
        y,
        t,
        spike_times,
        mask_zero_occupancy=True,
        interpolate_invalid=False,
        **kwargs
    ):
        """Calculate rate map as spike_map / occupancy_map
        Parameters
        ----------
        mask_zero_occupancy : bool
            Set pixels of zero occupancy to nan
        interpolate_invalid : bool
            Interpolate rate_map after division to remove invalid values,
            if False, and mask_zero_occupancy is False,
            invalid values are set to zero.
        kwargs : key word arguments to scipy.interpolate, when smoothing > 0
        Returns
        -------
        rate_map : array
        """
        spike_map = self.spike_map(
            x, y, t, spike_times, mask_zero_occupancy=mask_zero_occupancy, **kwargs
        )
        # to avoid infinity (x/0) we set zero occupancy to nan
        occupancy_map = self.occupancy_map(x, y, t, mask_zero_occupancy=True, **kwargs)
        rate_map = spike_map / occupancy_map
        if not mask_zero_occupancy:
            rate_map[np.isnan(rate_map)] = 0
        elif interpolate_invalid:
            rate_map = interpolate_nan_2D(rate_map)
        return rate_map
