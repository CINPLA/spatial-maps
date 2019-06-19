import numpy as np
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from .tools import fftcorrelate2d, autocorrelation


def find_peaks(image):
    """
    Find peaks sorted by distance from center of image.
    Returns
    -------
    peaks : array
        coordinates for peaks in image as [row, column]
    """
    image = image.copy()
    image[~np.isfinite(image)] = 0
    image_max = filters.maximum_filter(image, 3)
    is_maxima = (image == image_max)
    labels, num_objects = ndimage.label(is_maxima)
    indices = np.arange(1, num_objects+1)
    peaks = ndimage.maximum_position(image, labels=labels, index=indices)
    peaks = np.array(peaks)
    center = np.array(image.shape) / 2
    distances = np.linalg.norm(peaks - center, axis=1)
    peaks = peaks[distances.argsort()]
    return peaks


def separate_fields_by_laplace(rate_map, threshold=0):
    """Separates fields using the laplacian to identify fields separated by
    a negative second derivative.
    Parameters
    ----------
    rate_map : np 2d array
        firing rate in each bin
    threshold : float
        value of laplacian to separate fields by relative to the minima. Should be
        on the interval 0 to 1, where 0 cuts off at 0 and 1 cuts off at
        min(laplace(rate_map)). Default 0.
    Returns
    -------
    labels : numpy array, shape like rate_map.
        contains areas all filled with same value, corresponding to fields
        in rate_map. The fill values are in range(1,nFields + 1), sorted by size of the
        field (sum of all field values) with 0 elsewhere.
    field_count : int
        field count
    """
    from scipy import ndimage

    l = ndimage.laplace(rate_map)

    l[l>threshold*np.min(l)] = 0

    # Labels areas of the laplacian not connected by values > 0.
    fields, field_count = ndimage.label(l)

    # index 0 is the background
    indx = 1 + np.arange(field_count)

    # Sort by largest peak
    rate_means = ndimage.labeled_comprehension(
        rate_map, fields, indx, np.max, np.float64, 0)
    sort = np.argsort(rate_means)[::-1]

    # new rate map with fields > min_size, sorted
    new = np.zeros_like(fields)
    for i in range(field_count):
        new[fields == sort[i]+1] = i+1

    fields = new

    return fields


def calculate_field_centers(rate_map, labels, center_method='maxima'):
    """Finds center of fields at labels."""

    from scipy import ndimage
    indices = np.arange(1, np.max(labels) + 1)
    if center_method == 'maxima':
        bc = ndimage.maximum_position(
            rate_map, labels=labels, index=indices)
    elif center_method == 'center_of_mass':
        bc = ndimage.center_of_mass(
            rate_map, labels=labels, index=indices)
    else:
        raise ValueError(
            "invalid center_method flag '{}'".format(center_method))
    bc = np.array(bc)
    bc[:,[0, 1]] = bc[:,[1, 0]] # y, x -> x, y
    return bc


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
    [1]: Geoffrey W. Diehl, Olivia J. Hon, Stefan Leutgeb, Jill K. Leutgeb,
    https://doi.org/10.1016/j.neuron.2017.03.004
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
        borders = np.logical_and(fields > 0, wall)
        extents = labeled_comprehension(
            input=borders, labels=fields, index=None, func=np.sum,
            out_dtype=np.int64, default=0)
        max_extent = np.max([max_extent, np.max(extents)])

        # dont rotate the fourth time
        wall = np.rot90(wall) if i < 3 else wall

    C_M = max_extent / rate_map.shape[0]

    x = np.linspace(-0.5, 0.5, rate_map.shape[1])
    y = np.linspace(-0.5, 0.5, rate_map.shape[0])
    X,Y = np.meshgrid(x,y)

    # create linear increasing value towards middle
    dist_to_nearest_wall = 1 - (np.abs(X + Y) + np.abs(X - Y))

    d_m = np.average(dist_to_nearest_wall, weights=rate_map)
    b = (C_M - d_m) / (C_M + d_m)
    return b


def in_field(x, y, field, box_size):
    """Returns which spatial field each (x,y)-position is in. 

    Parameters:
    -----------
    x : numpy array
    y : numpy array, len(y) == len(x)
    field : numpy nd array 
        labeled fields, where each field is defined by an area separated by
        zeros. The fields are labeled with indices from [1:].
    box_size: list of two floats
        extents of arena

    Returns:
    --------
    indices : numpy array, length = len(x)
        arraylike x and y with field-labeled indices
    """

    if len(x)!= len(y):
        raise ValueError('x and y must have same length')

    sx,sy   = field.shape
    # bin sizes
    dx      = box_size[0]/sx
    dy      = box_size[1]/sy
    x_bins  = dx + np.arange(0, box_size[0], dx) 
    y_bins  = dy + np.arange(0, box_size[1], dy) 
    ix      = np.digitize(x, x_bins) 
    iy      = np.digitize(y, y_bins)

    # fix for boundaries:
    ix[ix==sx] = sx-1
    iy[iy==sy] = sy-1
    return np.array(field[ix,iy])
