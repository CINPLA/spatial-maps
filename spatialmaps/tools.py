import numpy as np


def autocorrelation(rate_map, mode='full', normalize=True):
    return fftcorrelate2d(rate_map, rate_map, mode=mode, normalize=normalize)
    

def fftcorrelate2d(arr1, arr2, mode='full', normalize=False):
    """
    Cross correlation of two 2 dimensional arrays using fftconvolve from scipy.
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
    return amp * np.exp(- 0.5 * (((x - xc) / s)**2 + ((y - yc) / s)**2))


def gaussian2D_asym(pos, amplitude, xc, yc, sigma_x, sigma_y, theta):
    x,y = pos

    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = amplitude*np.exp( - (a*((x-xc)**2) + 2*b*(x-xc)*(y-yc)
                            + c*((y-yc)**2)))
    return g.ravel()


def fit_gauss_asym(data, p0 = None, return_data=True):
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
        ia =     np.max(data)                                # amplitude
        index = np.unravel_index(np.argmax(data), (sx, sy))  # center
        ix, iy = x[index], y[index]
        isig =   0.01
        iang = 0

        p0 = (ia, ix, iy, isig, isig, iang)

    popt, pcov = curve_fit(gaussian2D_asym, (x, y), data.ravel(), p0=p0)
    # TODO : Add test for pcov
    if return_data:
        data_fitted = gaussian2D_asym((x, y), *popt)
        return popt, data_fitted.reshape(sx,sy)
    else:
        return popt
