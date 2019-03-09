from spatialmaps.tools import gaussian2D
import quantities as pq
import numpy as np


def make_test_grid_rate_map(
    box_size, bin_size, sigma=0.05, spacing=0.3, amplitude=1., offset=0):
    if isinstance(sigma, float):
        sigma = sigma * np.ones(7)
    if isinstance(amplitude, (float, int)):
        amplitude = amplitude * np.ones(7)
    xbins = np.arange(0, box_size[0], bin_size[0])
    ybins = np.arange(0, box_size[1], bin_size[1])
    x,y = np.meshgrid(xbins, ybins)

    p0 = np.array((box_size[0] / 2, box_size[1] / 2)) + offset
    pos = [p0]

    angles = np.linspace(0, 2 * np.pi, 7)[:-1]

    rate_map = np.zeros_like(x)
    rate_map += gaussian2D(amplitude[0], x, y, *p0, sigma[0])

    for i, a in enumerate(angles):
        p = p0 + [spacing * f(a) for f in [np.cos, np.sin]]
        rate_map += gaussian2D(amplitude[i+1], x, y, *p, sigma[i+1])
        pos.append(p)
    return rate_map, np.array(pos), xbins, ybins


def random_walk(box_size, step_size, n_step):
    # edited from https://stackoverflow.com/questions/48777345/vectorized-random-walk-in-python-with-boundaries
    start = np.array([0, 0])
    directions = np.array([(i,j) for i in [-1,0,1] for j in [-1,0,1]])
    boundaries = np.array([(0, box_size[0]), (0, box_size[1])])
    size = np.diff(boundaries, axis=1).ravel()
    # "simulation"
    trajectory = np.cumsum(
        directions[np.random.randint(0, 9, (n_step,))] * step_size, axis=0)
    trajectory = (
        np.abs((trajectory + start - boundaries[:, 0] + size) % (2 * size) - size)
        + boundaries[:, 0])
    return trajectory


def make_test_spike_map(
    pos_fields, box_size, bin_size, rate, n_step=10**6, step_size=.01):
    from elephant.spike_train_generation import homogeneous_poisson_process
    from scipy.interpolate import interp1d

    def infield(pos, pos_fields, sigma=.1):
        dist = np.sqrt(np.sum((pos - pos_fields)**2, axis=1))
        if any(dist <= sigma):
            return True
        else:
            return False

    dt = step_size / 1.5 # s / max_speed
    t = np.linspace(0, n_step * dt, n_step)
    trajectory = random_walk(box_size, step_size, n_step)
    x, y = trajectory.T
    st = homogeneous_poisson_process(
        rate=rate / pq.s, t_start=0 * pq.s, t_stop=t[-1] * pq.s).magnitude

    spike_pos = np.array([interp1d(t, x)(st), interp1d(t, y)(st)])

    spikes = [times for times, pos in zip(st, spike_pos.T)
              if infield(pos, pos_fields)]

    return np.array(x), np.array(y), np.array(t), np.array(spikes)
