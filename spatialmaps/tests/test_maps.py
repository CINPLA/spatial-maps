import numpy as np
import pytest
from spatialmaps import SpatialMap
from spatialmaps.tools import gaussian2D
from spatialmaps.maps import _digitize, _adjust_bin_size
import matplotlib.pyplot as plt

def make_test_grid_rate_map(
    box_size, bin_size, sigma=0.05, spacing=0.3, amplitude=1., dpos=0):
    if isinstance(sigma, float):
        sigma = sigma * np.ones(7)
    if isinstance(amplitude, (float, int)):
        amplitude = amplitude * np.ones(7)
    x = np.arange(0, box_size[0], bin_size[0])
    y = np.arange(0, box_size[1], bin_size[1])
    x,y = np.meshgrid(x,y)

    p0 = np.array((0.5, 0.5)) + dpos
    pos = [p0]

    angles = np.linspace(0, 2 * np.pi, 7)[:-1]

    rate_map = np.zeros_like(x)
    rate_map += gaussian2D(amplitude[0], x, y, *p0, sigma[0])

    for i, a in enumerate(angles):
        p = p0 + [spacing * f(a) for f in [np.cos, np.sin]]
        rate_map += gaussian2D(amplitude[i+1], x, y, *p, sigma[i+1])
        pos.append(p)
    return rate_map, np.array(pos)


def make_test_spike_map(rate_map, box_size, bin_size, n_step=10**6, step_size=.01):
    rate_bool = rate_map > .1
    ny, nx = rate_map.shape

    rate = 10

    dt = step_size / 1.5 # s / max_speed
    time = np.linspace(0, n_step * dt, n_step)
    # random walk from https://stackoverflow.com/questions/48777345/vectorized-random-walk-in-python-with-boundaries
    start = np.array([0, 0])
    directions = np.delete(np.indices((3, 3)).reshape(2, -1), 4, axis=1).T - 1
    boundaries = np.array([(0, box_size[0]), (0, box_size[1])])
    start = np.array([0, 0])

    # "simulation"
    size = np.diff(boundaries, axis=1).ravel()

    trajectory = np.cumsum(
        directions[np.random.randint(0, 8, (n_step,))] * step_size, axis=0)
    trajectory = (
        np.abs((trajectory + start - boundaries[:, 0] + size) % (2 * size) - size)
        + boundaries[:, 0])
    x, y = trajectory.T

    from elephant.spike_train_generation import homogeneous_poisson_process
    spike_times = homogeneous_poisson_process(
        rate=rate / pq.s, t_start=0 * pq.s, t_stop=time[-1] * pq.s)
    xbins, ybins, ix, iy = _digitize(x, y, box_size, bin_size)
    # spike_map = _spike_map(x, y, t, spike_times, xbins, ybins, ix, iy)
    # spike_map = spike_map[rate_bool]
    X, Y = np.meshgrid(xbins, ybins)
    xin = X[rate_bool] # TODO

    return np.array(x), np.array(y), np.array(time), np.array(spikes)


# def plot_path(x, y, t, box_size, spike_times=None,
#               color='grey', alpha=0.5, origin='upper',
#               spike_color='r', rate_markersize=False, markersize=10.,
#               animate=False, ax=None):
#     if ax is None:
#         fig = plt.figure()
#         ax = fig.add_subplot(
#             111, xlim=[0, box_size], ylim=[0, box_size], aspect=1)
#
#     ax.plot(x, y, c=color, alpha=alpha)
#     if spike_times is not None:
#         spikes_in_bin, _ = np.histogram(spike_times, t)
#         is_spikes_in_bin = spikes_in_bin > 0
#
#         if rate_markersize:
#             markersize = spikes_in_bin[is_spikes_in_bin] * markersize
#         ax.scatter(x[:-1][is_spikes_in_bin], y[:-1][is_spikes_in_bin],
#                    facecolor=spike_color, edgecolor=spike_color,
#                    s=markersize)
#
#     ax.grid(False)
#     if origin == 'upper':
#         ax.invert_yaxis()
#     return ax
#
#
# def test_rate_map():
#     box_size = 1.
#     rate = 10.
#     bin_size = .01
#     rate_map_true, pos = make_test_grid_rate_map(
#         sigma=0.05, spacing=0.3, amplitude=rate, dpos=0, box_size=box_size,
#         bin_size=bin_size)
#     rate_map_flat = rate_map_true.copy()
#     rate_map_flat[rate_map_flat > .1] = rate
#     x, y, t, spikes = make_test_grid_spike_path(
#         rate_map_flat, n_step=10**5, step_size=.02, box_size=1)
#     map = SpatialMap(
#         x, y, t, spikes, box_size=box_size, bin_size=bin_size)
#     rate_map = map.rate_map(.03)
#     plt.figure()
#     plt.imshow(rate_map)
#     plt.colorbar()
#     plt.figure()
#     plt.imshow(rate_map_true)
#     plt.colorbar()
#     plt.figure()
#     plt.imshow(rate_map_true - rate_map)
#     plt.colorbar()
#     plot_path(x, y, t, box_size, spikes)
#     plt.show()


def test_spatial_rate_map_diag():
    N = 10
    bin_size = 1
    box_size = 1.0
    x = np.linspace(0., box_size, N)
    y = np.linspace(0., box_size, N)
    t = np.linspace(0.1, 10.1, N)
    sptr = np.arange(0.1, 10.1, .5)
    map = SpatialMap(
        x, y, t, sptr, box_size=box_size, bin_size=bin_size)
    ratemap = map.rate_map(0)
    print(ratemap)
    assert all(np.diff(np.diag(ratemap)) < 1e-10)
    assert ratemap.shape == (int(box_size / bin_size), int(box_size / bin_size))


def test_occupancy_map_diag():
    N = 3
    bin_size = .5
    box_size = 1.5
    x = np.linspace(0., box_size, N)
    y = np.linspace(0., box_size, N)
    t = np.linspace(0, 10., N)

    map = SpatialMap(
        x, y, t, [], box_size=box_size, bin_size=bin_size)
    occmap_expected = np.array([[5, 0, 0],
                                [0, 5, 0],
                                [0, 0, 5]])
    occmap = map.occupancy_map(0)
    assert np.array_equal(occmap, occmap_expected)


if __name__ == '__main__':
    test_rate_map()
