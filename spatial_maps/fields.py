import numpy as np
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.interpolate import interp2d, interp1d
from .tools import fftcorrelate2d, autocorrelation

def border_score(rate_map, fields):
    raise DeprecationWarning('This function is moved to "spatial_maps.bordercells"')
    return spatial_maps.stats(rate_map, fields)


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
    center = (np.array(image.shape) - 1) / 2
    distances = np.linalg.norm(peaks - center, axis=1)
    peaks = peaks[distances.argsort()]
    return peaks


def sort_fields_by_rate(rate_map, fields, func=None):
    '''Sort fields by the rate value of each field
    Parameters
    ----------
    rate_map : array
        The rate map
    fields : array
        The fields same shape as rate_map
    func : function returning value to sort after, default np.max
    Returns
    -------
    sorted_fields : array
        Sorted fields
    '''
    indx = np.sort(np.unique(fields.ravel()))
    func = func or np.max
    # Sort by largest peak
    rate_means = ndimage.labeled_comprehension(
        rate_map, fields, indx, func, np.float64, 0)
    sort = np.argsort(rate_means)[::-1]

    # new rate map with fields > min_size, sorted
    sorted_fields = np.zeros_like(fields)
    for i in range(indx.max() + 1):
        sorted_fields[fields == sort[i] + 1] = i + 1

    return sorted_fields


def remove_fields_by_area(fields, minimum_field_area):
    '''Sets fields below minimum area to zero, measured as the number of bins in a field.
    Parameters
    ----------
    fields : array
        The fields
    minimum_field_area : int
        Minimum field area (number of bins in a field)
    Returns
    -------
    fields
        Fields with number of bins below minimum_field_area are set to zero
    '''
    if not isinstance(minimum_field_area, (int, np.integer)):
        raise ValueError("'minimum_field_area' should be int")

    ## variant
    # fields_areas = scipy.ndimage.measurements.sum(
    #     np.ones_like(fields), fields, index=np.arange(fields.max() + 1))
    # fields_area = fields_areas[fields]
    # fields[fields_area < minimum_field_area] = 0

    labels, counts = np.unique(fields, return_counts=True)
    for (lab, count) in zip(labels, counts):
        if lab != 0:
            if count < minimum_field_area:
                fields[fields == lab] = 0
    return fields


def separate_fields_by_laplace(rate_map, threshold=0, minimum_field_area=None):
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
    minimum_field_area: int
        minimum number of bins to consider it a field. Default None (all fields are kept)
    Returns
    -------
    labels : numpy array, shape like rate_map.
        contains areas all filled with same value, corresponding to fields
        in rate_map. The fill values are in range(1,nFields + 1), sorted by size of the
        field (sum of all field values) with 0 elsewhere.
    :Authors:
        Halvard Sutterud <halvard.sutterud@gmail.com>
    """

    l = ndimage.laplace(rate_map)

    l[l > threshold * np.min(l)] = 0

    # Labels areas of the laplacian not connected by values > 0.
    fields, field_count = ndimage.label(l)
    fields = sort_fields_by_rate(rate_map, fields)
    if minimum_field_area is not None:
        fields = remove_fields_by_area(fields, minimum_field_area)
    return fields


def separate_fields_by_dilation(rate_map, seed=2.5, sigma=2.5, minimum_field_area=None):
    """Separates fields by the Laplace of Gaussian (LoG)
    on the rate map subtracted by a reconstruction of the rate map using
    dilation.
    Parameters
    ----------
    rate_map : np 2d array
        firing rate in each bin
    seed : float
        Magnitude of dilation
    sigma : float
        Standard deviation of Gaussian to separate fields Default 2.
    minimum_field_area: int
        minimum number of bins to consider it a field. Default None (all fields are kept)
    Returns
    -------
    labels : numpy array, shape like rate_map.
        contains areas all filled with same value, corresponding to fields
        in rate_map. The fill values are in range(1,nFields + 1), sorted by size of the
        field (sum of all field values) with 0 elsewhere.
    References
    ----------
    see https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_regional_maxima.html
    """
    from skimage.morphology import reconstruction
    rate_map_norm = (rate_map - rate_map.mean()) / rate_map.std()
    dilated = reconstruction(rate_map_norm - seed, rate_map_norm, method='dilation')
    rate_map_reconstructed = rate_map_norm - dilated

    l = ndimage.gaussian_laplace(rate_map_reconstructed, sigma)
    l[l > 0] = 0
    fields, field_count = ndimage.label(l)
    fields = sort_fields_by_rate(rate_map, fields)
    if minimum_field_area is not None:
        fields = remove_fields_by_area(fields, minimum_field_area)
    return fields


def separate_fields_by_laplace_of_gaussian(rate_map, sigma=2, minimum_field_area=None):
    """Separates fields using the Laplace of Gaussian (LoG) to identify fields
    separated by a negative second derivative. Works best if no smoothing is
    applied to the rate map, preferably with interpolated nans.
    Parameters
    ----------
    rate_map : np 2d array
        firing rate in each bin
    sigma : float
        Standard deviation of Gaussian to separate fields Default 2.
    minimum_field_area: int
        minimum number of bins to consider it a field. Default None (all fields are kept)
    Returns
    -------
    labels : numpy array, shape like rate_map.
        contains areas all filled with same value, corresponding to fields
        in rate_map. The fill values are in range(1,nFields + 1), sorted by size of the
        field (sum of all field values) with 0 elsewhere.
    """
    l = ndimage.gaussian_laplace(rate_map, sigma)
    l[l > 0] = 0

    # Labels areas of the laplacian not connected by values > 0.
    fields, field_count = ndimage.label(l)

    fields = sort_fields_by_rate(rate_map, fields)
    if minimum_field_area is not None:
        fields = remove_fields_by_area(fields, minimum_field_area)
    return fields


def calculate_field_centers(rate_map, labels, center_method='maxima'):
    """Finds center of fields at labels.
    :Authors:
        Halvard Sutterud <halvard.sutterud@gmail.com>
    """

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
    if not bc:
        # empty list
        return bc
    bc = np.array(bc)
    bc[:,[0, 1]] = bc[:,[1, 0]] # y, x -> x, y
    return bc


def which_field(x, y, fields, box_size):
    """Returns which spatial field each (x,y)-position is in.

    Parameters:
    -----------
    x : numpy array
    y : numpy array, len(y) == len(x)
    fields : numpy nd array
        labeled fields, where each field is defined by an area separated by
        zeros. The fields are labeled with indices from [1:].
    box_size: list of two floats
        extents of arena

    Returns:
    --------
    indices : numpy array, length = len(x)
        arraylike x and y with fields-labeled indices
    """

    if len(x)!= len(y):
        raise ValueError('x and y must have same length')

    sx, sy = fields.shape
    # bin sizes
    dx = box_size[0]/sx
    dy = box_size[1]/sy
    x_bins = dx + np.arange(0, box_size[0] + dx, dx)
    y_bins = dy + np.arange(0, box_size[1] + dx, dy)
    # x_bins = np.arange(0, box_size[0] + dx, dx)
    # y_bins = np.arange(0, box_size[1] + dx, dy)
    ix = np.digitize(x, x_bins)
    iy = np.digitize(y, y_bins)

    # fix for boundaries:
    ix[ix==sx] = sx-1
    iy[iy==sy] = sy-1
    return np.array(fields[ix,iy])


def compute_crossings(field_indices):
    """Compute indices at which a field is entered or exited
    Parameters
    ----------
    field_indices : 1D array
        typically obtained with in_field
    See also
    --------
    in_field
    """
    # make sure to start and end outside fields
    field_indices = np.concatenate(([0], field_indices.astype(bool).astype(int), [0]))
    enter, = np.where(np.diff(field_indices) == 1)
    exit, = np.where(np.diff(field_indices) == -1)
    assert len(enter) == len(exit), (len(enter), len(exit))
    return enter, exit


def distance_to_edge_function(x_c, y_c, field, box_size, interpolation='linear'):
    """Returns a function which for a given angle returns the distance to
    the edge of the field from the center.
    Parameters:
        x_c
        y_c
        field: numpy 2d array
            ones at field bins, zero elsewhere
    """

    from skimage import measure
    contours = measure.find_contours(field, 0.8)

    box_dim = np.array(box_size)
    edge_x, edge_y = (contours[0] * box_dim / (np.array(field.shape) - (1, 1))).T

    # # angle between 0 and 2\pi
    angles = np.arctan2((edge_y - y_c), (edge_x - x_c)) % (2 * np.pi)
    a_sort = np.argsort(angles)
    angles = angles[a_sort]
    edge_x = edge_x[a_sort]
    edge_y = edge_y[a_sort]

    # # Fill in edge values for the interpolation
    pad_a = np.pad(angles, 2, mode='linear_ramp', end_values=(0, 2 * np.pi))
    ev_x = (edge_x[0] + edge_x[-1]) / 2
    pad_x = np.pad(edge_x, 2, mode='linear_ramp', end_values=ev_x)
    ev_y = (edge_y[0] + edge_y[-1]) / 2
    pad_y = np.pad(edge_y, 2, mode='linear_ramp', end_values=ev_y)

    if interpolation=='cubic':
        mask = np.where(np.diff(pad_a) == 0)
        pad_a = np.delete(pad_a, mask)
        pad_x = np.delete(pad_x, mask)
        pad_y = np.delete(pad_y, mask)

    x_func = interp1d(pad_a, pad_x, kind=interpolation)
    y_func = interp1d(pad_a, pad_y, kind=interpolation)

    def dist_func(angle):
        x = x_func(angle)
        y = y_func(angle)
        dist = np.sqrt((x - x_c)**2 + (y - y_c)**2)
        return dist

    return dist_func


def map_pass_to_unit_circle(x, y, t, x_c, y_c, field=None, box_size=None, dist_func=None):
    """Uses three vectors {v,p,q} to map the passes to the unit circle. v
    is the average velocity vector of the pass, p is the vector from the
    position (x,y) to the center of the field and q is the vector from the
    center to the edge through (x,y). See [1].

    Parameters:
    -----------
        :x, y, t: np arrays
            should contain x,y and t data in numpy arrays
        :x_c , y_c: floats
            bump center
        :field: np 2d array (optional)
            bins indicating location of field.
        :dist_func: function (optional)
            dist_func(angle) = distance to bump edge from center
            default is distance_to_edge_function with linear interpolation
        :return_vecs(optional): bool, default False

    See also:
    ---------
    distance_to_edge_function

    Returns:
    --------
        r : array of distance to origin on unit circle
        theta : array of angles to axis defined by mean velocity vector
        pdcd : array of distance to peak projected onto the current direction
        pdmd : array of distance to peak projected onto the mean direction

    References:
    -----------
        [1]: A. Jeewajee et. al., Theta phase precession of grid and
        placecell firing in open environment
    """
    if dist_func is None:
        assert field is not None and box_size is not None, (
            'either provide "dist_func" or "field" and "box_size"')
        dist_func= distance_to_edge_function(
            x_c, y_c, field, box_size, interpolation='linear')
    pos = np.array((x, y))

    # vector from pos to center p
    p_vec = ((x_c, y_c) - pos.T).T
    # angle between x-axis and negative vector p
    angle = (np.arctan2(p_vec[1], p_vec[0]) + np.pi) % (2 * np.pi)
    # distance from center to edge at each angle
    q = dist_func(angle)
    # distance from center to pos
    p = np.linalg.norm(p_vec, axis=0)
    # r-coordinate on unit circle
    r = p / q

    dpos = np.gradient(pos, axis=1)
    dt = np.gradient(t)
    velocity = np.divide(dpos, dt)

    # mean velocity vector v
    mean_velocity = np.average(velocity, axis=1)
    # angle on unit circle, run is rotated such that mean velocity vector
    # is toward positive x
    theta = (angle - np.arctan2(mean_velocity[1], mean_velocity[0])) % (2 * np.pi)

    w_pdcd = (angle - np.arctan2(velocity[1], velocity[0]))
    pdcd = r * np.cos(w_pdcd)

    w_pdmd = (angle - np.arctan2(mean_velocity[1], mean_velocity[0]))
    pdmd = r * np.cos(w_pdmd)
    return r, theta, pdcd, pdmd
