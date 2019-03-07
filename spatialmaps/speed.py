import numpy as np
import quantities as pq
import neo
import elephant as ep
# import exana.tracking as tr

def speed_correlation(speed, t, sptr, stddev = 250*pq.ms, min_speed=0.0*pq.m*pq.Hz,
                      max_speed_percentile=100, return_data=False):
    """
    Correlates instantaneous spike rate and rat velocity, using a method
    described in [1]
    Parameters:
    -----------
    speed, t :
    sptr :
    Returns
    -------

    out : correlation, (pearson, inst_speed)
    [1]:
    """

    from scipy.interpolate import interp1d
    max_speed = np.percentile(speed, max_speed_percentile)
    mask = np.logical_and(speed < max_speed, speed > min_speed)

    t = t * pq.s

    t_ = t[:speed.size][mask]
    speed = speed[mask]

    dt = np.average(np.diff(t_))

    interp_speed = interp1d(t_,speed,  bounds_error=False, fill_value = (speed[0], speed[-1]))
    binsize = np.mean(np.diff(t)).rescale('s') * 2

    rate = ep.statistics.instantaneous_rate(sptr, binsize, t_start = t[0], t_stop = t[-1],
                            kernel = ep.kernels.GaussianKernel(sigma=stddev))

    mask = np.logical_and(rate.times <= t[-1], rate.times >= t[0])
    inst_speed = interp_speed(rate.times[mask].rescale('1/Hz'))
    rate = rate.magnitude[mask,0]
    # rate = inst_rate.magnitude[:,0]
    correlation = np.corrcoef(inst_speed, rate)[1,0]

    return correlation, inst_speed, rate

