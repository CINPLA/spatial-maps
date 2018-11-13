import numpy as np


def spatial_rate_map(x, y, t, spike_train, binsize=0.01, box_xlen=1,
                     box_ylen=1, mask_unvisited=True, convolve=True,
                     return_bins=False, smoothing=0.02):
    """Divide a 2D space in bins of size binsize**2, count the number of spikes
    in each bin and divide by the time spent in respective bins. The map can
    then be convolved with a gaussian kernel of size csize determined by the
    smoothing factor, binsize and box_xlen.

    Parameters
    ----------
    spike_train : neo.SpikeTrain
    x : float
        1d vector of x positions
    y : float
        1d vector of y positions
    t : float
        1d vector of times at x, y positions
    binsize : float
        spatial binsize
    box_xlen : quantities scalar in m
        side length of quadratic box
    mask_unvisited: bool
        mask bins which has not been visited by nans
    convolve : bool
        convolve the rate map with a 2D Gaussian kernel

    Returns
    -------
    out : rate map
    if return_bins = True
    out : rate map, xbins, ybins
    """
    if not all([len(var) == len(var2) for var in [x,y,t] for var2 in [x,y,t]]):
        raise ValueError('x, y, t must have same number of elements')
    if box_xlen < x.max() or box_ylen < y.max():
        raise ValueError('box length must be larger or equal to max path length')
    from decimal import Decimal as dec
    decimals = 1e10
    remainderx = dec(float(box_xlen)*decimals) % dec(float(binsize)*decimals)
    remaindery = dec(float(box_ylen)*decimals) % dec(float(binsize)*decimals)
    if remainderx != 0 or remaindery != 0:
        raise ValueError('the remainder should be zero i.e. the ' +
                         'box length should be an exact multiple ' +
                         'of the binsize')

    # interpolate one extra timepoint
    t_ = np.append(t, t[-1] + np.median(np.diff(t)))
    spikes_in_bin, _ = np.histogram(spike_train, t_)
    time_in_bin = np.diff(t_)
    xbins = np.arange(0, box_xlen + binsize, binsize)
    ybins = np.arange(0, box_ylen + binsize, binsize)
    ix = np.digitize(x, xbins, right=True)
    iy = np.digitize(y, ybins, right=True)
    spike_pos = np.zeros((xbins.size, ybins.size))
    time_pos = np.zeros((xbins.size, ybins.size))
    for n in range(len(x)):
        spike_pos[ix[n], iy[n]] += spikes_in_bin[n]
        time_pos[ix[n], iy[n]] += time_in_bin[n]
    # correct for shifting of map
    spike_pos = spike_pos[1:, 1:]
    time_pos = time_pos[1:, 1:]
    with np.errstate(divide='ignore', invalid='ignore'):
        rate = np.divide(spike_pos, time_pos)
    if convolve:
        rate[np.isnan(rate)] = 0.  # for convolution
        from astropy.convolution import Gaussian2DKernel, convolve_fft
        csize = (box_xlen / binsize) * smoothing
        kernel = Gaussian2DKernel(csize)
        rate = convolve_fft(rate, kernel)  # TODO edge correction
    if mask_unvisited:
        was_in_bin = np.asarray(time_pos, dtype=bool)
        rate[np.invert(was_in_bin)] = np.nan
    if return_bins:
        return rate.T, xbins, ybins
    else:
        return rate.T


def gridness(rate_map, box_xlen, box_ylen, return_acorr=False,
             step_size=0.1, method='iter', return_masked_acorr=False):
    '''Calculates gridness of a rate map. Calculates the normalized
    autocorrelation (A) of a rate map B where A is given as
    A = 1/n\Sum_{x,y}(B - \bar{B})^{2}/\sigma_{B}^{2}. Further, the Pearsson's
    product-moment correlation coefficients is calculated between A and A_{rot}
    rotated 30 and 60 degrees. Finally the gridness is calculated as the
    difference between the minimum of coefficients at 60 degrees and the
    maximum of coefficients at 30 degrees i.e. gridness = min(r60) - max(r30).

    If the method 'iter' is chosen:
    In order to focus the analysis on symmetry of A the the central and the
    outer part of the gridness is maximized by increasingly mask A at steps of
    ``step_size``.

    If the method 'puncture' is chosen:
    This is the standard way of calculating gridness, by masking the central
    autocorrelation bump, in addition to rounding the map. See examples.

    Parameters
    ----------
    rate_map : numpy.ndarray
    box_xlen : float
        side length of quadratic box
    step_size : float
        step size in masking, only applies to the method "iter"
    return_acorr : bool
        return autocorrelation map or not
    return_masked_acorr : bool
        return masked autocorrelation map or not
    method : 'iter' or 'puncture'

    Returns
    -------
    out : gridness, (autocorrelation map, masked autocorrelation map)

    Examples
    --------
    >>> from exana.tracking.tools import make_test_grid_rate_map
    >>> import matplotlib.pyplot as plt
    >>> rate_map, pos = make_test_grid_rate_map()
    >>> iter_score = gridness(rate_map, box_xlen=1, box_ylen=1, method='iter')
    >>> print('%.2f' % iter_score)
    1.39
    >>> puncture_score = gridness(rate_map, box_xlen=1, box_ylen=1, method='puncture')
    >>> print('%.2f' % puncture_score)
    0.96

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from exana.tracking.tools import make_test_grid_rate_map
        from exana.tracking import gridness
        import matplotlib.pyplot as plt
        rate_map, _ = make_test_grid_rate_map()
        fig, axs = plt.subplots(2, 2)
        g1, acorr, m_acorr1 = gridness(rate_map, box_xlen=1,
                                         box_ylen=1, return_acorr=True,
                                         return_masked_acorr=True,
                                         method='iter')
        g2, m_acorr2 = gridness(rate_map, box_xlen=1,
                                         box_ylen=1,
                                         return_masked_acorr=True,
                                         method='puncture')
        mats = [rate_map, m_acorr1, acorr, m_acorr2]
        titles = ['Rate map', 'Masked acorr "iter", gridness = %.2f' % g1,
                  'Autocorrelation',
                  'Masked acorr "puncture", gridness = %.2f' % g2]
        for ax, mat, title in zip(axs.ravel(), mats, titles):
            ax.imshow(mat)
            ax.set_title(title)
        plt.tight_layout()
        plt.show()
    '''
    import numpy.ma as ma
    from exana.misc.tools import fftcorrelate2d
    from exana.tracking.tools import gaussian2D
    from scipy.optimize import curve_fit

    tmp_map = rate_map.copy()
    tmp_map[~np.isfinite(tmp_map)] = 0
    acorr = fftcorrelate2d(tmp_map, tmp_map, mode='full', normalize=True)
    rows, cols = acorr.shape
    b_x = np.linspace(- box_xlen / 2., box_xlen / 2., rows)
    b_y = np.linspace(- box_ylen / 2., box_ylen / 2., cols)
    B_x, B_y = np.meshgrid(b_x, b_y)
    if method == 'iter':
        if return_masked_acorr: m_acorrs = []
        gridscores = []
        for outer in np.arange(box_xlen / 4, box_xlen / 2, step_size):
            m_acorr = ma.masked_array(
                acorr, mask=np.sqrt(B_x**2 + B_y**2) > outer)
            for inner in np.arange(0, box_xlen / 4, step_size):
                m_acorr = ma.masked_array(
                    m_acorr, mask=np.sqrt(B_x**2 + B_y**2) < inner)
                r30, r60 = rotate_corr(m_acorr)
                gridscores.append(np.min(r60) - np.max(r30))
                if return_masked_acorr: m_acorrs.append(m_acorr)
        gridscore = max(gridscores)
        if return_masked_acorr: m_acorr = m_acorrs[gridscores.index(gridscore)]
    elif method == 'puncture':
        # round picture edges
        _gaussian = lambda pos, a, s: gaussian2D(a, pos[0], pos[1], 0, 0, s).ravel()
        p0 = (max(acorr.ravel()), min(box_xlen, box_ylen) / 100)
        popt, pcov = curve_fit(_gaussian, (B_x, B_y), acorr.ravel(), p0=p0)
        m_acorr = ma.masked_array(
            acorr, mask=np.sqrt(B_x**2 + B_y**2) > min(box_xlen, box_ylen) / 2)
        m_acorr = ma.masked_array(
            m_acorr, mask=np.sqrt(B_x**2 + B_y**2) < popt[1])
        r30, r60 = rotate_corr(m_acorr)
        gridscore = float(np.min(r60) - np.max(r30))
    if return_acorr and return_masked_acorr:
        return gridscore, acorr, m_acorr
    if return_masked_acorr:
        return gridscore, m_acorr
    if return_acorr:
        return gridscore, acorr  # acorrs[grids.index(max(grids))]
    else:
        return gridscore


def rotate_corr(acorr):
    from exana.misc.tools import masked_corrcoef2d
    from scipy.ndimage.interpolation import rotate
    angles = range(30, 180+30, 30)
    corr = []
    # Rotate and compute correlation coefficient
    for angle in angles:
        rot_acorr = rotate(acorr, angle, reshape=False)
        corr.append(masked_corrcoef2d(rot_acorr, acorr)[0, 1])
    r60 = corr[1::2]
    r30 = corr[::2]
    return r30, r60


def occupancy_map(x, y, t,
                  binsize=0.01,
                  box_xlen=1,
                  box_ylen=1,
                  mask_unvisited=True,
                  convolve=True,
                  return_bins=False,
                  smoothing=0.02):
    '''Divide a 2D space in bins of size binsize**2, count the time spent
    in each bin. The map can  be convolved with a gaussian kernel of size
    csize determined by the smoothing factor, binsize and box_xlen.

    Parameters
    ----------
    x : array
        1d vector of x positions
    y : array
        1d vector of y positions
    t : array
        1d vector of times at x, y positions
    binsize : float
        spatial binsize
    box_xlen : float
        side length of quadratic box
    mask_unvisited: bool
        mask bins which has not been visited by nans
    convolve : bool
        convolve the rate map with a 2D Gaussian kernel


    Returns
    -------
    occupancy_map : numpy.ndarray
    if return_bins = True
    out : occupancy_map, xbins, ybins
    '''

    if not all([len(var) == len(var2) for var in [
            x, y, t] for var2 in [x, y, t]]):
        raise ValueError('x, y, t must have same number of elements')
    if box_xlen < x.max() or box_ylen < y.max():
        raise ValueError(
            'box length must be larger or equal to max path length')
    from decimal import Decimal as dec
    decimals = 1e10
    remainderx = dec(float(box_xlen)*decimals) % dec(float(binsize)*decimals)
    remaindery = dec(float(box_ylen)*decimals) % dec(float(binsize)*decimals)
    if remainderx != 0 or remaindery != 0:
        raise ValueError('the remainder should be zero i.e. the ' +
                         'box length should be an exact multiple ' +
                         'of the binsize')

    # interpolate one extra timepoint
    t_ = np.array(t.tolist() + [t.max() + np.median(np.diff(t))])
    time_in_bin = np.diff(t_)
    xbins = np.arange(0, box_xlen + binsize, binsize)
    ybins = np.arange(0, box_ylen + binsize, binsize)
    ix = np.digitize(x, xbins, right=True)
    iy = np.digitize(y, ybins, right=True)
    time_pos = np.zeros((xbins.size, ybins.size))
    for n in range(len(x) - 1):
        time_pos[ix[n], iy[n]] += time_in_bin[n]
    # correct for shifting of map since digitize returns values at right edges
    time_pos = time_pos[1:, 1:]
    if convolve:
        rate[np.isnan(rate)] = 0.  # for convolution
        from astropy.convolution import Gaussian2DKernel, convolve_fft
        csize = (box_xlen / binsize) * smoothing
        kernel = Gaussian2DKernel(csize)
        rate = convolve_fft(rate, kernel)  # TODO edge correction
    if mask_unvisited:
        was_in_bin = np.asarray(time_pos, dtype=bool)
        rate[np.invert(was_in_bin)] = np.nan
    if return_bins:
        return rate.T, xbins, ybins
    else:
        return rate.T


def nvisits_map(x, y, t,
                binsize=0.01,
                box_xlen=1,
                box_ylen=1,
                return_bins=False):
    '''Divide a 2D space in bins of size binsize**2, count the
    number of visits in each bin. The map can  be convolved with
    a gaussian kernel of size  determined by the smoothing factor,
    binsize and box_xlen.

    Parameters
    ----------
    x : array
        1d vector of x positions
    y : array
        1d vector of y positions
    t : array
        1d vector of times at x, y positions
    binsize : float
        spatial binsize
    box_xlen : float
        side length of quadratic box


    Returns
    -------
    nvisits_map : numpy.ndarray
    if return_bins = True
    out : nvisits_map, xbins, ybins
    '''

    if not all([len(var) == len(var2) for var in [
            x, y, t] for var2 in [x, y, t]]):
        raise ValueError('x, y, t must have same number of elements')
    if box_xlen < x.max() or box_ylen < y.max():
        raise ValueError(
            'box length must be larger or equal to max path length')
    from decimal import Decimal as dec
    decimals = 1e10
    remainderx = dec(float(box_xlen)*decimals) % dec(float(binsize)*decimals)
    remaindery = dec(float(box_ylen)*decimals) % dec(float(binsize)*decimals)
    if remainderx != 0 or remaindery != 0:
        raise ValueError('the remainder should be zero i.e. the ' +
                         'box length should be an exact multiple ' +
                         'of the binsize')

    xbins = np.arange(0, box_xlen + binsize, binsize)
    ybins = np.arange(0, box_ylen + binsize, binsize)
    ix = np.digitize(x, xbins, right=True)
    iy = np.digitize(y, ybins, right=True)
    nvisits_map = np.zeros((xbins.size, ybins.size))
    for n in range(len(x)):
        if n == 0:
            nvisits_map[ix[n], iy[n]] = 1
        else:
            if ix[n-1] != ix[n] or iy[n-1] != iy[n]:
                nvisits_map[ix[n], iy[n]] += 1
    # correct for shifting of map since digitize returns values at right edges
    nvisits_map = nvisits_map[1:, 1:]
    if return_bins:
        return nvisits_map.T, xbins, ybins
    else:
        return nvisits_map.T


def spatial_rate_map_1d(x, t, spike_train,
                        binsize=0.01,
                        track_len=1,
                        mask_unvisited=True,
                        convolve=True,
                        return_bins=False,
                        smoothing=0.02):
    """Take x coordinates of linear track data, divide in bins of binsize,
    count the number of spikes  in each bin and  divide by the time spent in
    respective bins. The map can then be convolved with a gaussian kernel of
    size csize determined by the smoothing factor, binsize and box_xlen.

    Parameters
    ----------
    spike_train : array
    x : array
        1d vector of x positions
    t : array
        1d vector of times at x, y positions
    binsize : float
        spatial binsize
    box_xlen : float
        side length of quadratic box
    mask_unvisited: bool
        mask bins which has not been visited by nans
    convolve : bool
        convolve the rate map with a 2D Gaussian kernel

    Returns
    -------
    out : rate map
    if return_bins = True
    out : rate map, xbins
    """
    if not all([len(var) == len(var2) for var in [x, t] for var2 in [x, t]]):
        raise ValueError('x, t must have same number of elements')
    if track_len < x.max():
        raise ValueError('track length must be\
        larger or equal to max path length')
    from decimal import Decimal as dec
    decimals = 1e10
    remainderx = dec(float(track_len)*decimals) % dec(float(binsize)*decimals)
    if remainderx != 0:
        raise ValueError('the remainder should be zero i.e. the ' +
                         'box length should be an exact multiple ' +
                         'of the binsize')
    # interpolate one extra timepoint
    t_ = np.array(t.tolist() + [t.max() + np.median(np.diff(t))])
    spikes_in_bin, _ = np.histogram(spike_train, t_)
    time_in_bin = np.diff(t_)
    xbins = np.arange(0, track_len + binsize, binsize)
    ix = np.digitize(x, xbins, right=True)
    spike_pos = np.zeros(xbins.size)
    time_pos = np.zeros(xbins.size)
    for n in range(len(x)):
        spike_pos[ix[n]] += spikes_in_bin[n]
        time_pos[ix[n]] += time_in_bin[n]
    # correct for shifting of map since digitize returns values at right edges
    spike_pos = spike_pos[1:]
    time_pos = time_pos[1:]
    with np.errstate(divide='ignore', invalid='ignore'):
        rate = np.divide(spike_pos, time_pos)
    if convolve:
        rate[np.isnan(rate)] = 0.  # for convolution
        from astropy.convolution import Gaussian2DKernel, convolve_fft
        csize = (track_len / binsize) * smoothing
        kernel = Gaussian2DKernel(csize)
        rate = convolve_fft(rate, kernel)  # TODO edge correction
    if mask_unvisited:
        was_in_bin = np.asarray(time_pos, dtype=bool)
        rate[np.invert(was_in_bin)] = np.nan
    if return_bins:
        return rate.T, xbins
    else:
        return rate.T


def separate_fields(rate_map, laplace_thrsh=0, center_method='maxima',
        cutoff_method='none', box_xlen=1, box_ylen=1, index=False):
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
    center_method : string
        method to find field centers. Valid options = ['center_of_mass',
        'maxima','gaussian_fit']
    cutoff_method (optional) : string or function
        function to exclude small fields. If local field value of function
        is lower than global function value, the field is excluded. Valid
        string_options = ['median', 'mean','none'].
    index : bool, default False
        return bump center values as index or xy-pos

    Returns
    -------
    fields : numpy array, shape like rate_map.
        contains areas all filled with same value, corresponding to fields
        in rate_map. The values are in range(1,nFields + 1), sorted by size of the
        field (sum of all field values). 0 elsewhere.
    n_field : int
        field count
    bump_centers : (n_field x 2) np ndarray
        Coordinates of field centers
    """


    cutoff_functions = {'mean':np.mean, 'median':np.median, 'none':None}
    if not callable(cutoff_method):
        try:
            cutoff_func = cutoff_functions[cutoff_method]
        except KeyError:
            msg = "invalid cutoff_method flag '%s'" % cutoff_method
            raise ValueError(msg)
    else:
        cutoff_func = cutoff_method

    from scipy import ndimage

    l = ndimage.laplace(rate_map)

    l[l>laplace_thrsh*np.min(l)] = 0

    # Labels areas of the laplacian not connected by values > 0.
    fields, n_fields = ndimage.label(l)

    # index 0 is the background
    indx = np.arange(1,n_fields+1)

    # Use cutoff method to remove unwanted fields
    if cutoff_method != 'none':
        try:
            total_value = cutoff_func(fields)
        except:
            print('Unexpected error, cutoff_func doesnt like the input:')
            raise

        field_values = ndimage.labeled_comprehension(rate_map, fields, indx,
                cutoff_func, float, 0)
        try:
            is_field = field_values >= total_value
        except:
            print('cutoff_func return_values doesnt want to compare:')
            raise

        if np.sum(is_field) == 0:
            return np.zeros(rate_map.shape), 0, np.array([[],[]])

        for i in indx:
            if not is_field[i-1]:
                fields[fields == i] = 0


        n_fields = ndimage.label(fields, output=fields)
        indx = np.arange(1,n_fields + 1)

    # Sort by largest mean
    sizes = ndimage.labeled_comprehension(rate_map, fields, indx,
            np.mean, float, 0)
    size_sort = np.argsort(sizes)[::-1]
    new = np.zeros_like(fields)
    for i in np.arange(n_fields):
        new[fields == size_sort[i]+1] = i+1
    fields = new

    bc = get_bump_centers(rate_map,labels=fields,ret_index=index,indices=indx,method=center_method,
                          units=box_xlen.units)

    # TODO exclude fields where maxima is on the edge of the field?
    return fields, n_fields, bc


def get_bump_centers(rate_map, labels, ret_index=False, indices=None, method='maxima',
        units=1):
    """Finds center of fields at labels."""

    from scipy import ndimage

    if method not in ['maxima','center_of_mass','gaussian_fit']:
        msg = "invalid center_method flag '%s'" % method
        raise ValueError(msg)
    if indices is None:
        indices = np.arange(1,np.max(labels)+1)
    if method == 'maxima':
        bc = ndimage.maximum_position(rate_map, labels=labels,
                index=indices)
    elif method == 'center_of_mass':
        bc = ndimage.center_of_mass(rate_map, labels=labels, index=indices)
    elif method == 'gaussian_fit':
        from  exana.tracking.tools import fit_gauss_asym
        bc = np.zeros((len(indices),2))
        import matplotlib.pyplot as plt
        for i in indices:
            r = rate_map.copy()
            r[labels != i] = 0
            popt = fit_gauss_asym(r, return_data=False)
            # TODO Find out which axis is x and which is y
            bc[i-1] = (popt[2],popt[1])
        if ret_index:
            msg = 'ret_index not implemented for gaussian fit'
            raise NotImplementedError(msg)
    if not ret_index and not method=='gaussian_fit':
        bc = (bc + np.array((0.5,0.5)))/rate_map.shape
    return np.array(bc)*units



def find_avg_dist(rate_map, thrsh = 0, plot=False):
    """Uses autocorrelation and separate_fields to find average distance
    between bumps. Is dependent on high gridness to get separate bumps in
    the autocorrelation

    Parameters
    ----------
    rate_map : np 2d array
               firing rate in each bin

    thrsh (optional) : float, default 0
        cutoff value for the laplacian of the autocorrelation function.
        Should be a negative number. Gives better separation if bumps are
        connected by "bridges" or saddles where the laplacian is negative.
    plot (optional) : bool, default False
        plot acorr and the separated acorr, with bump centers
    Returns
    -------
    avg_dist : float
        relative units from 0 to 1 of the box size
        """

    from scipy.ndimage import maximum_position
    from exana.misc.tools import fftcorrelate2d

    # autocorrelate. Returns array (2x - 1) the size of rate_map
    acorr = fftcorrelate2d(rate_map,rate_map, mode = 'full', normalize = True)

    #acorr[acorr<0] = 0 # TODO Fix this
    f, nf, bump_centers = separate_fields(acorr,laplace_thrsh=thrsh,
            center_method='maxima',cutoff_method='median')
                                         # TODO Find a way to find valid value for
                                         # thrsh, or remove.

    bump_centers = np.array(bump_centers)

    # find dists from center in (autocorrelation)relative units (from 0 to 1)
    distances = np.linalg.norm(bump_centers - (0.5,0.5), axis = 1)

    dist_sort = np.argsort(distances)
    distances = distances[dist_sort]

    # use maximum 6 closest values except center value
    avg_dist = np.median(distances[1:7])

    # correct for difference in shapes
    avg_dist *= acorr.shape[0]/rate_map.shape[0] # = 1.98


    # TODO : raise warning if too big difference between points
    if plot:
        import matplotlib.pyplot as plt
        fig,[ax1,ax2] = plt.subplots(1,2)

        ax1.imshow(acorr,extent  = (0,1,0,1),origin='lower')
        ax1.scatter(*(bump_centers[:,::-1].T))
        ax2.imshow(f,extent  = (0,1,0,1),origin='lower')
        ax2.scatter(*(bump_centers[:,::-1].T))
    return avg_dist


def fit_hex(bump_centers, avg_dist=None, plot_bumps = False, method='best'):
    """Fits a hex grid to a given set of bumps. Uses the three bumps most


    Parameters
    ----------
    bump_centers : Nx2 np.array
                x,y positions of bump centers, x,y /in (0,1)

    avg_dist (optional): float
                average spacing between bumps

    plot_bumps (optional): bool
                if True, plots at the three bumps most likely to be in
                correct hex-position to the current matplotlib axes.

    method (optional): string, valid options: ['closest', 'best']
                method to find angle from neighboring bumps.
                'closest' uses six bumps nearest to center bump
                'best' uses the two bumps nearest to avg_dist

    Returns
    -------
    displacement : float
                   distance of bump closest to the center in meters
    orientation : float
                  orientation of hexagon (in degrees)
    """

    valid_methods = ['closest', 'best']
    if method not in valid_methods:
        msg = "invalid method flag '%s'" % method
        raise ValueError(msg)
    bump_centers = np.array(bump_centers)

    # sort by distance to center
    d = np.linalg.norm(bump_centers - (0.5,0.5), axis=1)
    d_sort = np.argsort(d)
    dist_sorted = bump_centers[d_sort]
    center_bump = dist_sorted[0]; others = dist_sorted[1:]

    displacement = d[d_sort][0]

    # others distances to center bumps
    relpos = others - center_bump
    reldist = np.linalg.norm(relpos, axis=1)

    if method == 'closest':
        # get 6 closest bumps
        rel_sort = np.argsort(reldist)
        closest = others[rel_sort][:6]
        relpos = relpos[rel_sort][:6]
    elif method == 'best':
        # get 2 bumps such that /sum_{i\neqj}(\abs{r_i-r_j}-avg_ist)^2 is minimized
        squares = 1e32*np.ones((others.shape[0], others.shape[0]))

        for i in range(len(relpos)):
            for j in range(i,len(relpos)):
                rel1 = (reldist[i] - avg_dist)**2
                rel2 = (reldist[j] - avg_dist)**2
                rel3 = (np.linalg.norm(relpos[i]-relpos[j]) - avg_dist)**2
                squares[i,j] = rel1 + rel2 + rel3
        rel_slice = np.unravel_index(np.argmin(squares), squares.shape)
        rel_slice = np.array(rel_slice)
        #rel_sort = np.argsort(np.abs(reldist-avg_dist))
        closest = others[rel_slice]
        relpos = relpos[rel_slice]

    # sort by angle
    a = np.arctan2(relpos[:,1], relpos[:,0])%(2*np.pi)
    a_sort = np.argsort(a)

    # extract lowest angle and convert to degrees
    orientation = a[a_sort][0] *180/np.pi

    # hex grid is symmetric under rotations of 60deg
    orientation %= 60

    if plot_bumps:
        import matplotlib.pyplot as plt
        ax=plt.gca()
        i = 1
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        dx = xmax-xmin; dy = ymax - ymin

        closest = closest[a_sort]

        edges = [center_bump] if method == 'best' else []
        edges += [c for c in closest]
        edges = np.array(edges)*(dx,dy) + (xmin, ymin)
        poly = plt.Polygon(edges, alpha=0.5,color='r')
        ax.add_artist(poly)
    return displacement, orientation


def calculate_grid_geometry(rate_map, plot_fields=False, **kwargs):
    """Calculates quantitative information about grid field.
    Find bump centers, bump spacing, center diplacement and hexagon
    orientation

    Parameters
    ----------
    rate_map : np 2d array
        firing rate in each bin
    plot_fields : if True, plots the field labels with field centers to the
        current matplotlib ax. Default False
    thrsh : float, default 0
        see find_avg_dist()
    center_method : string, valid options: ['maxima', 'center_of_mass']
        default: 'center_of_mass'
        see separate_fields()
    method : string, valid options: ['closest', 'best']
        see fit_hex()

    Returns
    -------
    bump_centers : 2d np.array
        x,y positions of bump centers
    avg_dist : float
        average spacing between bumps, \in [0,1]
    displacement : float
        distance of bump closest to the center
    orientation : float
        orientation of hexagon (in degrees)

    Examples
    --------
    >>> import numpy as np
    >>> rate_map = np.zeros((5,5))
    >>> pos = np.array([  [0,2],
    ...                [1,0],[1,4],
    ...                   [2,2],
    ...                [3,0],[3,4],
    ...                   [4,2]])
    >>> for(i,j) in pos:
    ...     rate_map[i,j] = 1
    ...
    >>> result = calculate_grid_geometry(rate_map)
    """

    # TODO add back the following when it is correct
    # (array([[0.5, 0.9],
           # [0.9, 0.7],
           # [0.1, 0.7],
           # [0.5, 0.5],
           # [0.9, 0.3],
           # [0.1, 0.3],
           # [0.5, 0.1]]) * m, 0.4472135954999579, 0.0, 26.565051177077983)

    from scipy.ndimage import mean, center_of_mass

    # TODO: smooth data?
    # smooth_rate_map = lambda x:x
    # rate_map = smooth_rate_map(rate_map)

    center_method = kwargs.pop('center_method',None)
    if center_method:
        fields, nfields, bump_centers = separate_fields(rate_map,
                                        center_method=center_method)
    else:
        fields, nfields, bump_centers = separate_fields(rate_map)

    if bump_centers.size == 0:
        import warnings
        msg = 'couldnt find bump centers, returning None'
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return None,None,None,None,

    sh = np.array(rate_map.shape)

    if plot_fields:
        print(fields)
        import matplotlib.pyplot as plt
        x=np.linspace(0,1,sh[0]+1)
        y=np.linspace(0,1,sh[1]+1)
        x,y = np.meshgrid(x,y)
        ax = plt.gca()
        print('nfields: ',nfields)
        plt.pcolormesh(x,y, fields)

    # switch from row-column to x-y
    bump_centers = bump_centers[:,::-1]

    thrsh = kwargs.pop('thrsh', None)
    if thrsh:
        avg_dist = find_avg_dist(rate_map, thrsh)
    else:
        avg_dist = find_avg_dist(rate_map)

    displacement, orientation = fit_hex(bump_centers, avg_dist,
            plot_bumps=plot_fields, **kwargs)

    return bump_centers, avg_dist, displacement, orientation


class RandomDisplacementBounds(object):
    """random displacement with bounds"""
    def __init__(self, xmin, xmax, stepsize=0.5):
        self.xmin = np.array(xmin)
        self.xmax = np.array(xmax)
        self.stepsize = stepsize

    def __call__(self, x):
        """take a random step but ensure the new position is within the bounds"""
        while True:

            # this could be done in a much more clever way, but it will work for example purposes
            xnew = x + (self.xmax-self.xmin)*np.random.uniform(-self.stepsize,
                                                               self.stepsize, np.shape(x))
            if np.all(xnew < self.xmax) and np.all(xnew > self.xmin):
                break
        return xnew



def optimize_sep_fields(rate_map,step = 0.04, niter=40, T = 1.0, method = 'SLSQP',
        glob=True, x0 = [0.065,0.1],callback=None):
    """Optimizes the separation of the fields by minimizing an error
    function

    Parameters
    ----------
    rate_map :
    method :
        valid methods=['L-BFGS-B', 'TNC', 'SLSQP']
    x0 : list
        initial values for smoothing smoothing and laplace_thrsh

    Returns
    --------
    res :
        Result of the optimization. Contains smoothing and laplace_thrsh in
        attribute res.x"""

    from scipy import optimize
    from exana.tracking.tools import separation_error_func as err_func

    valid_methods = ['L-BFGS-B', 'TNC', 'SLSQP']
    if method not in valid_methods:
        raise ValueError('invalid method flag %s' %method)

    rate_map[np.isnan(rate_map)] = 0.

    method = 'SLSQP'
    xmin = [0.025, 0]
    xmax = [0.2,  1]
    bounds = [(low,high) for low,high in zip(xmin,xmax)]

    obj_func = lambda args: err_func(args[0], args[1], rate_map)

    if glob:
        take_step = RandomDisplacementBounds(xmin, xmax,stepsize=step)
        minimizer_kwargs = dict(method=method, bounds=bounds)
        res = optimize.basinhopping(obj_func, x0, niter=niter, T = T,
                minimizer_kwargs=minimizer_kwargs,
                take_step=take_step,callback=callback)
    else:
        res = optimize.minimize(obj_func, x0, method=method, bounds = bounds, options={'disp': True})
    return res




if __name__ == "__main__":
    import doctest
    doctest.testmod()
