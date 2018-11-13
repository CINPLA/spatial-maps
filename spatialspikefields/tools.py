import numpy as np
from ..misc.tools import normalize
from .head import head_direction
import scipy.signal as sig


def unit_vector(v):
    """ Return unit vector of v
    modified from David Wolever,
    https://stackoverflow.com/questions/2827393/angles
    -between-two-n-dimensional-vectors-in-python
    """
    return v / np.linalg.norm(v)


def angle_between_vectors(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'
    modified from David Wolever,
    https://stackoverflow.com/questions/2827393/angles
    -between-two-n-dimensional-vectors-in-python
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def rescale_linear_track_2d_to_1d(x, y, end_0=[], end_1=[]):
    """ Take x, y coordinates of linear track data, rescale to 1-d.

    Parameters
    ----------
    x : quantities.Quantity array in m
        1d vector of x positions
    y : quantities.Quantity array in m
        1d vector of x positions
    t : quantities.Quantity array in s
        1d vector of times at x, y positions
    end_0: quantities.Quantity array in m
        linear track endpoint 1, in x, y
    end_1: quantities.Quantity array in m
        linear track endpoint 2, in x, y

    Returns
    -------
    out : 1d vector
    """
    if not all([len(var) == len(var2) for var in
                [x, y] for var2 in [x, y]]):
        raise ValueError('x, y, t must have same number of elements')
    # shift coordinate system to have end_0 as origin
    x -= end_0[0]
    y -= end_0[1]

    # calculate angle of track
    v_x_axis = np.array([1, 0])
    theta = angle_between_vectors(end_1-end_0, v_x_axis)
    # rotate clockwise
    rot_mat = np.array([[np.cos(-theta), -np.sin(-theta)],
                        [np.sin(-theta),  np.cos(-theta)]])
    x_rot = []
    for x_i, y_i in zip(x, y):
        [x_rot_i, _] = np.dot(rot_mat,
                              np.array([[x_i],
                                        [y_i]]))
        x_rot.append(x_rot_i.item())
    # shift x_rot so that np.min(x_rot) == 0
    x_rot -= np.min(x_rot)
    # only consider x_rot in output
    return x_rot


def find_laps(peaks_start, peaks_stop, valid_start, valid_stop):
    laps = []
    for t_start, x_start in peaks_start:
        # check if current peak is in valid start position
        if not valid_start[0] <= x_start <= valid_start[1]:
            continue
        # find next maxpeak in time
        res = np.where(peaks_stop[:, 0] > t_start)[0]
        if len(res) == 0:
            continue
        id_stop = res[0]
        t_stop = peaks_stop[id_stop, 0]
        x_stop = peaks_stop[id_stop, 1]
        # check if stop peak is in its valid start zone
        if not valid_stop[0] <= x_stop <= valid_stop[1]:
            continue
        # add start and end time of lap
        laps.append([[t_start, t_stop], [x_start, x_stop]])
    return laps


def gaussian2D(amp, x, y, xc, yc, s):
    return amp * np.exp(- 0.5 * (((x - xc) / s)**2 + ((y - yc) / s)**2))


def make_test_grid_rate_map(sigma=0.05, spacing=0.3,
                            amplitude=1., dpos=0,
                            box_xlen=1., box_ylen=1.):
    if isinstance(sigma, float):
        sigma = sigma * np.ones(7)
    if isinstance(amplitude, float):
        amplitude = amplitude * np.ones(7)
    x = np.linspace(0, box_xlen, 50)
    y = np.linspace(0, box_ylen, 50)
    x,y = np.meshgrid(x,y)

    p0 = np.array((0.5, 0.5)) + dpos
    pos = [p0]

    angles = np.linspace(0, 2 * np.pi, 7)[:-1]

    rate_map = np.zeros_like(x)
    rate_map += gaussian2D(1, x, y, *p0, sigma[0])

    for i, a in enumerate(angles):
        p = p0 + [spacing * f(a) for f in [np.cos, np.sin]]
        rate_map += gaussian2D(amplitude[i], x, y, *p, sigma[i])
        pos.append(p)
    return rate_map, np.array(pos)


def make_test_grid_spike_path(t_stop=600, dt=1/30, box_xlen=1, box_ylen=1):
    from elephant.spike_train_generation import homogeneous_poisson_process as hpp
    rate_map, grid_pos = make_test_grid_rate_map(box_xlen=box_xlen,
                                                 box_ylen=box_ylen)
    rate_map = rate_map > 0.1
    ny, nx = rate_map.shape
    xref = np.linspace(0, box_xlen, nx)
    yref = np.linspace(0, box_ylen, ny)
    time = np.arange(0, t_stop, dt)

    def speed_good(x1, y1, x2, y2, threshold=1.5):
        if any(x is None for x in [x1, y1, x2, y2]):
            return False
        return (np.sqrt((x2 - x1)**2 + (y2 - y1)**2)) / dt < threshold
    x, y, spikes = [0], [0], []
    while len(x) < len(time):
        x2, y2 = None, None
        while not speed_good(x[-1], y[-1], x2, y2):
            x2, y2 = np.random.uniform(0, 1, 2)
            x2, y2 = x2 * box_xlen, y2 * box_ylen
        x.append(x2)
        y.append(y2)
        if in_rate_map(rate_map, x2, y2, xref, yref):
            curr_t = time[len(x) - 1]
            st = hpp(rate=30.0, t_start=curr_t,
                     t_stop=(curr_t + dt))
            spikes.extend(st.times.magnitude.tolist())


    return x, y, time, spikes


def in_rate_map(rate_map, x, y, xref, yref):
    assert rate_map.dtype == bool
    xdiff = xref - x
    xdiff[xdiff < 0] = np.inf
    xidx = np.argmin(xdiff)
    ydiff = yref - y
    ydiff[ydiff < 0] = np.inf
    yidx = np.argmin(ydiff)
    return rate_map[yidx, xidx]


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


def separation_error_func(smoothing, lpl_thrsh, rate_map):
    """
    Gives a measure of how well the smoothing and laplace-threshold factors
    separates a rate_map into hexagonal fields.
    Measures the deviation of the distance from each bump to its two
    closest neighbors from the average distance as gotten from
    tr.fields.find_avg_dist, and the relative difference in area of each of
    the fields.

    Parameters
    -----------
        smoothing : float
            size of the smoothing kernel relative to the box
        lpl_thrsh : float

    laplace_thrsh : float
        value of laplacian to separate fields by relative to the minima.
        see exana.tracking.fields.separate_fields

    Returns
    -------
        err : float
            0 if all fields exact same size and distance from two closest
            neighbors
    """
    from astropy.convolution import Gaussian2DKernel, convolve_fft

    if np.isnan(smoothing):
        return np.inf

    import exana.tracking as tr

    rate_map[np.isnan(rate_map)] = 0.

    csize = rate_map.shape[0] * smoothing
    kernel = Gaussian2DKernel(csize)
    rm_smooth = convolve_fft(rate_map, kernel)  # TODO edge correction

    f, nf, bc = tr.fields.separate_fields(rm_smooth, laplace_thrsh=lpl_thrsh,
                                            cutoff_method = 'median',
                                            center_method = 'maxima')

    avg_dist = tr.fields.find_avg_dist(rm_smooth, thrsh = 0.1)

    if nf < 3:
        return np.inf
    if np.isnan(avg_dist):
        return np.inf

    indx = np.arange(1,nf+1)
    err = 0


    # Slower:
    # areas = np.zeros(nf)
    # for i in range(nf):
    #     areas[i] = np.sum(f==(i+1))
    # Faster: (~ 2x, depends on indx.size, bigger loop = more gain )
    areas = np.sum(f.ravel() == indx[:,None], axis = 1)

    # Slower:
    # area_deviation = 0
    # for i in range(nf):
    #     for j in range(i+1,nf):
    #         Ai = areas[i]
    #         Aj = areas[j]
    #         area_diff = (Ai - Aj)**2/(Ai*Aj)
    #         area_deviation += area_diff
    # Faster: (~ 4x)
    area_deviation = np.sum(((areas[:,None] - areas)**2/(areas[:,None] * areas)))
    err += area_deviation

    # Slower:
    # dist_deviations = np.zeros(nf)
    # for i in indx:
    #     bump = bc[i-1]

    #     rel = bc - bump
    #     dist = np.linalg.norm(rel, axis=1)
    #     sort = np.argsort(dist)

    #     # add relative difference in area to all other fields
    #     dist_diff = (dist[sort][1:3] - avg_dist)
    #     dist_deviations[i-1] = np.sum(dist_diff**2)/2
    # err += np.sum(dist_deviations**2)/nf
    # Faster (~ 4x)
    dists = np.linalg.norm(bc[:,None,:] - bc, axis = -1)
    dist_diffs = np.sort(dists)[:,1:3] - avg_dist
    dist_deviation = np.sum(np.sum(dist_diffs**2, axis = 1)**2)
    #err += dist_deviation
    # oneliner forzelulz
    # err += np.sum(np.sum((np.sort(np.linalg.norm(bc[:,None,:] - bc, axis = -1))[:,1:3] - avg)**2, axis = 1)**2)

    field_mask = f > 0

    # measure of the spike rates covered by the fields
    # TWO WAYS TO DO THIS: measure over original rate map, or over smoothed rate map
    #field_coverage =  np.sum(rm_smooth) / np.sum(rm_smooth[field_mask])
    field_coverage =  np.sum(rate_map) / np.sum(rate_map[field_mask])
    # total spike rate

    err = err*field_coverage/nf
    return err
