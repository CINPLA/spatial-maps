import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
from scipy.ndimage.measurements import center_of_mass


def plot_path(x, y, t, box_size, spike_times=None,
              color='grey', alpha=0.5, origin='upper',
              spike_color='r', rate_markersize=False, markersize=10.,
              animate=False, ax=None):
    """
    Plot path visited

    Parameters
    ----------
    x : array
        1d vector of x positions
    y : array
        1d vector of y positions
    t : array
        1d vector of time at x, y positions
    spike_times : array
    box_size : scalar
        size of spatial 2d square
    color : path color
    alpha : opacity of path
    spike_color : spike marker color
    rate_markersize : bool
        scale marker size to firing rate
    markersize : float
        size of spike marker
    animate : bool
    ax : matplotlib axes

    Returns
    -------
    out : ax
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(
            111, xlim=[0, box_size], ylim=[0, box_size], aspect=1)

    ax.plot(x, y, c=color, alpha=alpha)
    if spike_times is not None:
        spikes_in_bin, _ = np.histogram(spike_times, t)
        is_spikes_in_bin = spikes_in_bin > 0

        if rate_markersize:
            markersize = spikes_in_bin[is_spikes_in_bin] * markersize
        ax.scatter(x[:-1][is_spikes_in_bin], y[:-1][is_spikes_in_bin],
                   facecolor=spike_color, edgecolor=spike_color,
                   s=markersize)

    ax.grid(False)
    if origin == 'upper':
        ax.invert_yaxis()
    return ax


def animate_path(x, y, t, box_size, spike_times=None,
              color='grey', alpha=0.5, origin='upper',
              spike_color='r', rate_markersize=False, markersize=10.,
              animate=False, ax=None, title=''):
    """
    Plot path visited

    Parameters
    ----------
    x : array
        1d vector of x positions
    y : array
        1d vector of y positions
    t : array
        1d vector of time at x, y positions
    spike_times : array
    box_size : scalar
        size of spatial 2d square
    color : path color
    alpha : opacity of path
    spike_color : spike marker color
    rate_markersize : bool
        scale marker size to firing rate
    markersize : float
        size of spike marker
    animate : bool
    ax : matplotlib axes

    Returns
    -------
    out : ax
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(
            111, xlim=[0, box_size], ylim=[0, box_size], aspect=1)
    if spike_times is not None:
        spikes_in_bin, _ = np.histogram(spike_times, t)
        is_spikes_in_bin = np.array(spikes_in_bin, dtype=bool)

        if rate_markersize:
            markersizes = spikes_in_bin[is_spikes_in_bin]*markersize
        else:
            markersizes = markersize*np.ones(is_spikes_in_bin.size)
    ax.set_title(title)
    ax.grid(False)
    if origin == 'upper':
        ax.invert_yaxis()
    import time
    plt.show()
    for idx, x, y, active, msize in zip(range(len(x)), x, y):
        ax.plot(x, y, c=color, alpha=alpha)
        if spike_times is not None:
            if is_spikes_in_bin[idx]:
                ax.scatter(x, y, facecolor=spike_color, edgecolor=spike_color,
                           s=markersizes[idx])
        time.sleep(0.1)  # plt.pause(0.0001)
        plt.draw()
    return ax


def plot_head_direction_rate(spike_times, ang_bins, rate_in_ang, projection='polar',
                             normalization=False, ax=None, color='k'):
    """


    Parameters
    ----------
    spike_times : neo.SpikeTrain
    ang_bins : angular bin_size
        ang_bins must be in degrees
    rate_in_ang :
    projection : 'polar' or None
    normalization :
    group_name
    ax : matplotlib axes
    mask_unvisited : True: mask bins which has not been visited

    Returns
    -------
    out : ax
    """
    import math
    if normalization:
        rate_in_ang = normalize(rate_in_ang, mode='minmax')
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=projection)
    bin_size = ang_bins[1] - ang_bins[0]
    if projection is None:
        ax.set_xticks(range(0, 360 + 60, 60))
        ax.set_xlim(0, 360)
    elif projection == 'polar':
        ang_bins = [math.radians(deg) for deg in ang_bins]
        bin_size = math.radians(bin_size)
        ax.set_xticks([0, np.pi])
    ax.bar(ang_bins, rate_in_ang, width=bin_size, color=color)
    return ax


def plot_ratemap(x, y, t, spike_times, bin_size=0.05, box_size=1,
                 box_size=1, vmin=0, ax=None, smoothing=.05,
                 origin='upper', cmap='viridis'):
    """


    Parameters
    ----------
    x : 1d vector of x positions
    y : 1d vector of y positions
    t : 1d vector of time at x, y positions
    spike_times : array
    bin_size : size of spatial 2d square bins
    vmin : color min
    ax : matplotlib axes
    mask_unvisited : True: mask bins which has not been visited

    Returns
    -------
    out : axes
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, xlim=[0, 1], ylim=[0, 1], aspect=1)

    map = SpatialMap(
        x, y, t, spike_times, bin_size=bin_size, box_size=box_size)
    rate_map = map.rate_map(smoothing)
    ax.imshow(rate_map, interpolation='none', origin=origin,
              extent=(0, 1, 0, 1), vmin=vmin, cmap=cmap)
    ax.set_title('%.2f Hz' % np.nanmax(rate_map))
    ax.grid(False)
    return ax


def plot_occupancy(x, y, t, bin_size=0.05, box_size=1, box_size=1,
                  vmin=0, ax=None, convolve=True,
                  origin='upper', cmap='jet'):
    """


    Parameters
    ----------
    x : 1d vector of x positions
    y : 1d vector of y positions
    t : 1d vector of time at x, y positions
    spike_times : one neo.SpikeTrain
    bin_size : size of spatial 2d square bins
    vmin : color min
    ax : matplotlib axes
    mask_unvisited : True: mask bins which has not been visited

    Returns
    -------
    out : axes
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, xlim=[0, 1], ylim=[0, 1], aspect=1)

    occ_map = occupancy_map(x, y, t, bin_size=bin_size, box_size=box_size,
                             box_size=box_size, convolve=convolve)
    cax = ax.imshow(occ_map, interpolation='none', origin=origin,
                   extent=(0, 1, 0, 1), vmin=vmin, cmap=cmap, aspect='auto')
    # ax.set_title('%.2f s' % np.nanmax(occ_map))
    ax.grid(False)
    return cax, np.nanmax(occ_map)
