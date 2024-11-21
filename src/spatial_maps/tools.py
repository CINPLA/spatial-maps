import numpy as np


def autocorrelation(rate_map, mode="full", normalize=True):
    return fftcorrelate2d(rate_map, rate_map, mode=mode, normalize=normalize)


def fftcorrelate2d(arr1, arr2, mode="full", normalize=False):
    """
    Cross correlation of two 2 dimensional arrays using fftconvolve from scipy.
    Here we exploit the fact that correlation is convolution with one input
    rotated 180 degrees. See https://dsp.stackexchange.com/questions/12684/difference-between-correlation-and-convolution-on-an-image
    Parameters
    ----------
    arr1 : np.array
        2D array
    arr2 : np.array
        2D array
    mode : str
        Sent directly to numpe.fftconvolve
    normalize : bool
        Normalize arrays before convolution or not. Default is False.
    See also
    --------
    scipy.signal.fftconvolve : SciPy convolve function using fft.
    Returns
    -------
    corr : np.array
        Cross correlation
    Example
    --------
    >>> a = np.reshape(np.arange(4), (2,2))
    >>> acorr = fftcorrelate2d(a, a)
    """
    # TODO replace with astropy - just verify results are the same
    from scipy.signal import fftconvolve

    if normalize:
        a_ = np.reshape(arr1, (1, arr1.size))
        v_ = np.reshape(arr2, (1, arr2.size))
        arr1 = (arr1 - np.mean(a_)) / (np.std(a_) * len(a_))
        arr2 = (arr2 - np.mean(v_)) / np.std(v_)
    corr = fftconvolve(arr1, np.fliplr(np.flipud(arr2)), mode=mode)
    return corr


def nancorrelate2d(X, Y, mode="frobenius") -> np.ndarray:
    """
    Calculate 2d pearson correlation from matrices with nans interpreted as
    missing values, i.e. they are not included in any calculations. Also ignore
    values outside correlation windows of X and Y.

    Parameters
    ----------
    X : np.ndarray
        2D array, input.
    Y : np.ndarray
        2D array, kernel.
    mode : string
        either 'pearson' or 'frobenius' for window aggregations

    Returns
    -------
    Z : np.ndarray
        2D array (same shape as X and Y) of nan and border ignored spatial cross correlation

    Example
    -------
    >>> import numpy as np
    >>> X, Y = np.ones((2,2)), np.ones((2,2))
    >>> Z = nancorrelate2d(X, Y, mode='frobenius')
    >>> Z
    array([[1., 1.],
           [1., 1.]])
    """
    import numpy.ma as ma

    X = ma.masked_array(X, mask=np.isnan(X))
    Y = Y[::-1][:, ::-1]  #
    Y = ma.masked_array(Y, mask=np.isnan(Y))

    result = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            scope_i = slice(
                max(0, i - X.shape[0] // 2), min(i + X.shape[0] // 2, X.shape[0])
            )
            scope_j = slice(
                max(0, j - X.shape[1] // 2), min(j + X.shape[1] // 2, X.shape[1])
            )
            if mode == "pearson":
                result[i, j] = ma.corrcoef(
                    X[scope_i, scope_j].flatten(),
                    Y[scope_i, scope_j][::-1][:, ::-1].flatten(),
                )[0, 1]
            elif mode == "frobenius":  # scaled (average) frobenius inner product
                result[i, j] = (
                    X[scope_i, scope_j] * Y[scope_i, scope_j][::-1][:, ::-1]
                ).mean()
            else:
                raise NotImplementedError("Method does not have mode={}".format(mode))

    return result


def masked_corrcoef2d(arr1, arr2):
    """
    Correlation coefficient of two 2 dimensional masked arrays.
    Parameters
    ----------
    arr1 : np.array
        2D array.
    arr2 : np.array
        2D array.
    See also
    --------
    numpy.corrcoef : NumPy corrcoef function.
    numpy.ma : NumPy mask module.
    Returns
    -------
    corr : np.array
        correlation coefficient from np.corrcoef.
    Example
    --------
    >>> import numpy.ma as ma
    >>> a = np.reshape(np.arange(10), (2,5))
    >>> v = np.reshape(np.arange(10), (2,5))
    >>> mask = np.zeros((2, 5), dtype=bool)
    >>> mask[1:, 3:] = True
    >>> v = ma.masked_array(v, mask=mask)
    >>> print(v)
    [[0 1 2 3 4]
     [5 6 7 -- --]]
    >>> masked_corrcoef2d(a, v)
    masked_array(
      data=[[1.0, 1.0],
            [1.0, 1.0]],
      mask=[[False, False],
            [False, False]],
      fill_value=1e+20)
    """
    import numpy.ma as ma

    a_ = np.reshape(arr1, (1, arr1.size))
    v_ = np.reshape(arr2, (1, arr2.size))
    corr = ma.corrcoef(a_, v_)
    return corr


def gaussian2D(amp, x, y, xc, yc, s):
    return amp * np.exp(-0.5 * (((x - xc) / s) ** 2 + ((y - yc) / s) ** 2))


def gaussian2D_asym(pos, amplitude, xc, yc, sigma_x, sigma_y, theta):
    x, y = pos

    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (
        2 * sigma_y ** 2
    )
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (
        4 * sigma_y ** 2
    )
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (
        2 * sigma_y ** 2
    )
    g = amplitude * np.exp(
        -(a * ((x - xc) ** 2) + 2 * b * (x - xc) * (y - yc) + c * ((y - yc) ** 2))
    )
    return g.ravel()


def fit_gauss_asym(data, p0=None, return_data=True):
    """Fits an asymmetric 2D gauss function to the given data set, with optional guess
    parameters. Optimizes amplitude, center coordinates, sigmax, sigmay and
    angle. If no guess parameters, initializes with a thin gauss bell
    centered at the data maxima

    Parameters
    -----------
    data        : 2D np array
    p0 (optional): arraylike
                  initial parameters [amplitude,x_center,y_center,sigma_x, sigma_y,angle]
    return_data : bool



    Returns
    --------
    params      : tuple of params: (amp,xc,yc,sigmax, sigmay, angle)
    (if return_data) data_fitted : 2D np array
                                   the fitted gauss data

    """
    from scipy.optimize import curve_fit

    # Create x and y indices
    sx, sy = data.shape
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    x = np.linspace(xmin, xmax, sx)
    y = np.linspace(ymin, ymax, sy)
    x, y = np.meshgrid(x, y)

    if p0 is None:
        # initial guesses, use small gaussian at maxima as initial guess
        ia = np.max(data)  # amplitude
        index = np.unravel_index(np.argmax(data), (sx, sy))  # center
        ix, iy = x[index], y[index]
        isig = 0.01
        iang = 0

        p0 = (ia, ix, iy, isig, isig, iang)

    popt, pcov = curve_fit(gaussian2D_asym, (x, y), data.ravel(), p0=p0)
    # TODO : Add test for pcov
    if return_data:
        data_fitted = gaussian2D_asym((x, y), *popt)
        return popt, data_fitted.reshape(sx, sy)
    else:
        return popt


def stationary_poisson(t_start, t_stop, rate):
    """
    Stationary Poisson process
    Parameters
    ----------
    t_start : float
        Start time of the process (lower bound).
    t_stop : float
        Stop time of the process (upper bound).
    rate : float
        rate of the Poisson process
    Returns
    -------
    events : array
        time points from a Poisson process with rate rate.
    """
    n_exp = rate * (t_stop - t_start)
    return np.sort(np.random.uniform(t_start, t_stop, np.random.poisson(n_exp)))


def make_test_grid_rate_map(
    box_size,
    bin_size,
    sigma=0.05,
    spacing=0.3,
    amplitude=1.0,
    repeat=2,
    orientation=0,
    offset=0,
):
    box_size = np.array(box_size)
    xbins = np.arange(0, box_size[0], bin_size[0])
    ybins = np.arange(0, box_size[1], bin_size[1])
    x, y = np.meshgrid(xbins, ybins)

    p0 = np.array([box_size[0] / 2, box_size[1] / 2]) + offset
    pos = [p0]
    angles = np.linspace(0, 2 * np.pi, 7)[:-1] + orientation

    rate_map = np.zeros_like(x)
    rate_map += gaussian2D(amplitude, x, y, *p0, sigma)

    def add_hex(p0, pos, rate_map):
        for i, a in enumerate(angles):
            p = p0 + [spacing * f(a) for f in [np.cos, np.sin]]
            if not np.isclose(p, pos).prod(axis=1).any() and all(p <= box_size + sigma):
                rate_map += gaussian2D(amplitude, x, y, *p, sigma)
                pos = np.vstack((pos, p))
        return pos, rate_map

    pos, rate_map = add_hex(p0, pos, rate_map)
    for _ in range(repeat):
        pos_1 = pos.copy()
        for p1 in pos_1:
            pos, rate_map = add_hex(p1, pos, rate_map)

    return rate_map, np.array(pos), xbins, ybins


def make_test_border_map(box_size, bin_size, sigma=0.05, amplitude=1.0, offset=0):

    xbins = np.arange(0, box_size[0], bin_size[0])
    ybins = np.arange(0, box_size[1], bin_size[1])
    x, y = np.meshgrid(xbins, ybins)

    p0 = np.array((box_size[0], box_size[1] / 2)) + offset
    pos = [p0]

    angles = np.linspace(0, 2 * np.pi, 7)[:-1]

    rate_map = np.zeros_like(x)
    rate_map += gaussian2D(amplitude, x, y, *p0, sigma)

    return rate_map, np.array(pos), xbins, ybins


def random_walk(box_size, step_size, n_step, sampling_rate, low_pass=5):
    import scipy.signal as ss

    # edited from https://stackoverflow.com/questions/48777345/vectorized-random-walk-in-python-with-boundaries
    start = np.array([0, 0])
    directions = np.array([(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1]])
    boundaries = np.array([(0, box_size[0]), (0, box_size[1])])
    size = np.diff(boundaries, axis=1).ravel()
    # "simulation"
    trajectory = np.cumsum(
        directions[np.random.randint(0, 9, (n_step,))] * step_size, axis=0
    )
    x, y = (
        np.abs((trajectory + start - boundaries[:, 0] + size) % (2 * size) - size)
        + boundaries[:, 0]
    ).T

    b, a = ss.butter(N=1, Wn=low_pass * 2 / sampling_rate)
    # zero phase shift filter
    x = ss.filtfilt(b, a, x)
    y = ss.filtfilt(b, a, y)
    # we tolerate small interpolation errors
    x[(x > -1e-3) & (x < 0.0)] = 0.0
    y[(y > -1e-3) & (y < 0.0)] = 0.0

    return x, y


def make_test_spike_map(
    rate, sigma, pos_fields, box_size, n_step=10 ** 4, step_size=0.05
):
    from scipy.interpolate import interp1d

    def infield(pos, pos_fields):
        dist = np.sqrt(np.sum((pos - pos_fields) ** 2, axis=1))
        if any(dist <= sigma):
            return True
        else:
            return False

    t = np.linspace(0, n_step * step_size / 1.5, n_step)  # s / max_speed
    x, y = random_walk(box_size, step_size, n_step, sampling_rate=1 / (t[1] - t[0]))

    st = stationary_poisson(rate=rate, t_start=0, t_stop=t[-1])

    spike_pos = np.array([interp1d(t, x)(st), interp1d(t, y)(st)])

    spikes = [times for times, pos in zip(st, spike_pos.T) if infield(pos, pos_fields)]

    return np.array(x), np.array(y), np.array(t), np.array(spikes)
