import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve_fft
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from .tools import _fftcorrelate2d, _masked_corrcoef2d, autocorrelation


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


def _digitize(x, y, box_size, bin_size):
    xbins = np.arange(0, box_size[0] + bin_size[0], bin_size[0])
    ybins = np.arange(0, box_size[1] + bin_size[1], bin_size[1])

    ix = np.digitize(x, xbins, right=True)
    iy = np.digitize(y, ybins, right=True)

    return xbins, ybins, ix, iy


def rotate_corr(acorr, mask):
    import numpy.ma as ma
    from scipy.ndimage.interpolation import rotate
    m_acorr = ma.masked_array(acorr, mask=mask)
    angles = range(30, 180+30, 30)
    corr = []
    # Rotate and compute correlation coefficient
    for angle in angles:
        rot_acorr = rotate(acorr, angle, reshape=False)
        rot_acorr = ma.masked_array(rot_acorr, mask=mask)
        corr.append(_masked_corrcoef2d(rot_acorr, m_acorr)[0, 1])
    r60 = corr[1::2]
    r30 = corr[::2]
    return r30, r60

def find_peaks(image):
    """
    Returns peaks sorted by distance from center of image.
    """
    image_max = filters.maximum_filter(image, 3)
    is_maxima = (image == image_max)
    labels, num_objects = ndimage.label(is_maxima)
    indices = np.arange(1, num_objects+1)
    peaks = ndimage.maximum_position(image, labels=labels, index=indices)
    peaks = np.array(peaks)
    center = np.array(image.shape) / 2
    distances = np.linalg.norm(peaks - center, axis=1)
    peaks_sorted = peaks[distances.argsort()]
    return peaks_sorted

def peak_to_peak_distance(sorted_peaks, index_a, index_b):
    """
    Distance between peaks ordered by their distance from center.

    The peaks are first ordered.
    The distance between the requested peaks is calculated,
    where the indices determine which peak in the order is requested.

    If one of the peak indices does not exist, this function returns infinity.
    """
    try:
        distance = np.linalg.norm(sorted_peaks[index_b] - sorted_peaks[index_a])
    except IndexError:
        distance = np.inf
    return distance


def smooth_map(rate_map, bin_size, smoothing):
    std_dev_pixels = smoothing / bin_size
    rate_map = rate_map.copy()  # do not modify the original!
    rate_map[np.isnan(rate_map)] = 0.
    kernel = Gaussian2DKernel(std_dev_pixels[0], std_dev_pixels[1])
    return convolve_fft(rate_map, kernel)


def _occupancy_map(x, y, t, xbins, ybins, ix, iy):
    t_ = np.append(t, t[-1] + np.median(np.diff(t)))
    time_in_bin = np.diff(t_)
    values, _, _ = np.histogram2d(y, x, bins=[xbins, ybins], weights=time_in_bin)
    return values


def _spike_map(x, y, t, spike_times, xbins, ybins, ix, iy):
    t_ = np.append(t, t[-1] + np.median(np.diff(t)))
    spikes_in_bin, _ = np.histogram(spike_times, t_)
    values, _, _ = np.histogram2d(y, x, bins=[xbins, ybins], weights=spikes_in_bin)
    return values


def separate_fields(rate_map, laplace_thrsh=0):
    """Separates fields using the laplacian to identify fields separated by
    a negative second derivative.
    Parameters
    ----------
    rate_map : np 2d array
        firing rate in each bin
    laplace_thrsh : float
        value of laplacian to separate fields by relative to the minima. Should be
        on the interval 0 to 1, where 0 cuts off at 0 and 1 cuts off at
        min(laplace(rate_map)). Default 0.
    Returns
    -------
    fields : numpy array, shape like rate_map.
        contains areas all filled with same value, corresponding to fields
        in rate_map. The fill values are in range(1,nFields + 1), sorted by size of the
        field (sum of all field values) with 0 elsewhere.
    field_count : int
        field count
    """
    from scipy import ndimage

    l = ndimage.laplace(rate_map)

    l[l>laplace_thrsh*np.min(l)] = 0

    # Labels areas of the laplacian not connected by values > 0.
    fields, field_count = ndimage.label(l)

    # index 0 is the background
    indx = 1 + np.arange(field_count)

    # Sort by largest peak
    rate_means = ndimage.labeled_comprehension(rate_map, fields, indx, np.max, np.float64, 0)
    sort = np.argsort(rate_means)[::-1]

    # new rate map with fields > min_size, sorted
    new = np.zeros_like(fields)
    for i in range(field_count):
        new[fields == sort[i]+1] = i+1

    fields = new

    return fields, field_count


def border_score(rate_map, fields):
    """
    Uses a separation of the fields in a rate map to calculate a border
    score as described in [1].
    Parameters
    ----------
    rate_map : np 2d array
        firing rate in each bin
    fields : np 2d array of ints
        areas all filled with same value, corresponding to fields
        in rate_map. See output of separate_fields
    References
    ----------
    [1]: Geoffrey W. Diehl, Olivia J. Hon, Stefan Leutgeb, Jill K. Leutgeb, https://doi.org/10.1016/j.neuron.2017.03.004

    """

    from scipy.ndimage import labeled_comprehension

    # Find parts of fields next to border. Use a second to outer bins, as
    # outer bins won't be specified as field
    if np.all(fields == 0):
        raise ValueError("Must have at least one field")

    inner = np.zeros(np.array(rate_map.shape)-(4,4),dtype=bool)
    wall = np.pad(inner, 1, 'constant', constant_values=[[0,0],[1,0]])
    wall = np.pad(wall, 1, 'constant', constant_values=0)
    max_extent = 0

    ones = np.ones_like(rate_map)

    for i in range(4):
        borders = np.logical_and(fields>0, wall)
        extents = labeled_comprehension(
            input=borders, labels=fields, index=None, func=np.sum,
            out_dtype=np.int64, default=0)
        max_extent = np.max([max_extent, np.max(extents)])

        # dont rotate the fourth time
        wall = np.rot90(wall) if i < 3 else wall

    C_M = max_extent/rate_map.shape[0]

    x = np.linspace(-0.5,0.5,rate_map.shape[1])
    y = np.linspace(-0.5,0.5,rate_map.shape[0])
    X,Y = np.meshgrid(x,y)

    # create linear increasing value towards middle
    dist_to_nearest_wall = 1-(np.abs(X+Y)+np.abs(X-Y))

    d_m = np.average(dist_to_nearest_wall, weights=rate_map)
    b = (C_M - d_m)/(C_M + d_m)
    return b


def fields_from_distance(rate_map):
    import scipy.spatial as spatial

    acorr = autocorrelation(rate_map, mode='full', normalize=True)
    acorr_maxima = find_peaks(acorr)

    def place_field_radius(auto_correlation, maxima):
        map_size = np.array(auto_correlation.shape)
        center = map_size / 2
        distances = np.linalg.norm(maxima - center, axis=1)
        distances_sorted = sorted(distances)
        min_distance = distances_sorted[1] # the first one is basically the center
        # TODO consider a different factor than 0.7
        return 0.7 * min_distance / 2 # 0.7 because that is what Ismakov et al. used

    # TODO verify this for an example where there are fields too close
    def too_close_removed(rate_map, rate_map_maxima, place_field_radius):
        result = []
        rate_map_maxima_value = rate_map[tuple(rate_map_maxima.T)]
        distances = spatial.distance.cdist(rate_map_maxima, rate_map_maxima)
        too_close_pairs = np.where(distances < place_field_radius*2)
        not_accepted = []

        for i, j in zip(*too_close_pairs):
            if i == j:
                continue

            if rate_map_maxima_value[i] > rate_map_maxima_value[j]:
                not_accepted.append(j)
            else:
                not_accepted.append(i)

        for i in range(len(rate_map_maxima)):
            if i in not_accepted:
                continue

            result.append(rate_map_maxima[i])

        return np.array(result)

    radius = place_field_radius(acorr, acorr_maxima)

    rate_map_maxima = find_peaks(rate_map)
    rate_map_maxima = too_close_removed(rate_map, rate_map_maxima, radius)
    # rate_map_maxima = rate_map_maxima.astype(float) * bin_size
    # radius *= bin_size[0]

    return rate_map_maxima, radius

class SpatialMap:
    def __init__(self, x, y, t, spike_times, box_size, bin_size, bin_count=None):
        box_size, bin_size = _adjust_bin_size(box_size, bin_size, bin_count)
        xbins, ybins, ix, iy = _digitize(x, y, box_size, bin_size)

        self.spike_pos = _spike_map(x, y, t, spike_times, xbins, ybins, ix, iy)
        self.time_pos = _occupancy_map(x, y, t, xbins, ybins, ix, iy)

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

    def rate_map(self, smoothing):
        return self.spike_map(smoothing) / self.occupancy_map(smoothing)
