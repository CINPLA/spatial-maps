import numpy as np


def head_direction_rate(spike_train, head_angles, t, binsize=4, n_avg_bin=4):
    """
    Calculeate firing rate at head direction in binned head angles for time t.
    Moving average filter is applied on firing rate

    Parameters
    ----------
    spike_train : array
    head_angles : array in degrees
        all recorded head directions
    t : array
        1d vector of times at x, y positions
    binsize : float
        angular binsize
    n_avg_bin : int
        number of bins to average over

    Returns
    -------
    out : np.ndarray, np.ndarray
        binned angles, avg rate in corresponding bins
    """
    assert len(head_angles) == len(t)
    from ..misc.tools import moving_average
    # make bins around angle measurements
    spikes_in_bin, _ = np.histogram(spike_train, t)
    # take out the first and every other bin
    time_in_bin = np.diff(t)
    med_time = np.median(time_in_bin)
    # bin head_angles
    ang_bins = np.arange(0, 360 + binsize, binsize)
    ia = np.digitize(head_angles, ang_bins, right=True)
    spikes_in_ang = np.zeros(ang_bins.size)
    time_in_ang = np.zeros(ang_bins.size)
    for n in range(len(head_angles) - 1):
            # if time_in_bin[n] <= med_time:
            spikes_in_ang[ia[n]] += spikes_in_bin[n]
            time_in_ang[ia[n]] += time_in_bin[n]
    with np.errstate(divide='ignore', invalid='ignore'):
        rate_in_ang = np.divide(spikes_in_ang, time_in_ang)
    rate_in_ang = moving_average(rate_in_ang, n_avg_bin)
    return ang_bins, rate_in_ang


def head_direction_stats(head_angle_bins, rate):
    """
    Calculeate firing rate at head direction in head angles for time t

    Parameters
    ----------
    head_angle_bins : array in radians
        binned head directions
    rate : array
        firing rate magnitude coresponding to angles

    Returns
    -------
    out : float, float
        mean angle, mean vector length
    """
    import math
    import pycircstat as pc
    if any(np.isnan(rate)):
        raise ValueError('Nan not supported')
    head_angle_bins = np.delete(head_angle_bins, nanIndices)
    mean_ang = pc.mean(head_angle_bins, w=rate)
    mean_vec_len = pc.resultant_vector_length(head_angle_bins, w=rate)
    # ci_lim = pc.mean_ci_limits(head_angle_bins, w=rate)
    return mean_ang, mean_vec_len


def head_direction(x1, y1, x2, y2, t, filt=2.):
    """
    Calculeate head direction in angles or radians for time t

    Parameters
    ----------
    x1 : quantities.Quantity array in m
        1d vector of x positions from LED 1
    y1 : quantities.Quantity array in m
        1d vector of y positions from LED 1
    x2 : quantities.Quantity array in m
        1d vector of x positions from LED 2
    y2 : quantities.Quantity array in m
        1d vector of x positions from LED 2
    t : quantities.Quantity array in s
        1d vector of times from LED 1 or 2 at x, y positions
    filt : float
        threshold filter all LED distances less than filt*std(dist)

    Returns
    -------
    out : angles, resized t
    """
    import math
    measurements = len(x2)
    indeces = np.arange(measurements)
    # NaN elements:
    nanIndices = np.concatenate((np.where(np.isnan(x1))[0],
                                 np.where(np.isnan(x2))[0],
                                 np.where(np.isnan(y1))[0],
                                 np.where(np.isnan(y2))[0]))
    nanIndices = np.unique(nanIndices)
    x1 = np.delete(x1, nanIndices)
    y1 = np.delete(y1, nanIndices)
    x2 = np.delete(x2, nanIndices)
    y2 = np.delete(y2, nanIndices)
    indeces = np.delete(indeces, nanIndices)
    # Wrong length elements:
    dr = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    dr_mean = np.mean(dr)
    dr_std = np.std(dr)
    drIndices = np.where(dr < dr_mean - filt*dr_std)[0]
    drIndices = np.unique(drIndices)
    x1 = np.delete(x1, drIndices)
    y1 = np.delete(y1, drIndices)
    x2 = np.delete(x2, drIndices)
    y2 = np.delete(y2, drIndices)
    indeces = np.delete(indeces, drIndices)
    print("Removed %0.1f %% invalid measurements for head direction" %
          ((len(nanIndices)+len(drIndices))/float(measurements) * 100.))

    # Calculate angles in range [0, 2pi]:

    dx_g = x2 - x1
    dy_g = y2 - y1

    angles_rad = np.arctan2(dy_g, dx_g)
    tmpIndices = np.where(angles_rad < 0)
    angles_rad[tmpIndices] += 2 * np.pi
    return angles_rad, t[indeces]
