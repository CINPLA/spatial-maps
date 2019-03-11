import numpy as np
import pytest
import quantities as pq
import neo
from tools import random_walk
from spatialmaps.speed import speed_correlation
# TODO need a better test, where we can set the corr beforehand

def test_speed_random():
    box_size = [1., 1.]
    rate = 5.
    bin_size = [.01, .01]
    n_step=10**4
    step_size=.01

    from elephant.spike_train_generation import homogeneous_poisson_process
    from scipy.interpolate import interp1d

    t = np.linspace(0, n_step * step_size / 1.5, n_step)
    dt = t[1]
    trajectory = random_walk(box_size, step_size, n_step)
    x, y = trajectory.T
    st = homogeneous_poisson_process(
        rate=rate / pq.s, t_start=0 * pq.s, t_stop=t[-1] * pq.s)
    s = np.sqrt(x*x + y*y)
    speed = np.diff(s) / dt
    corr, inst_speed, st_rate = speed_correlation(speed, t, st)
    assert  np.abs(corr) < 0.05
    assert st_rate.mean().round() == rate


def test_speed_linear():
    box_size = [1., 1.]
    n_step=10**3

    from scipy.interpolate import interp1d

    t = np.linspace(0, n_step * .1, n_step)
    speed = np.linspace(0, 5, n_step)
    dt = 1
    s = 0
    spikes = []
    while dt > .001:
        dt /= 1.01
        next_s = s + dt
        spikes.append(next_s)
        s = next_s
    spikes = neo.SpikeTrain(spikes, t_stop=round(spikes[-1]), units='s')

    corr, inst_speed, st_rate = speed_correlation(speed, t, spikes)
    assert  round(corr, 2) == 0.36
