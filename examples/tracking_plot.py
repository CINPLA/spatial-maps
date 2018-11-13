import neo
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import quantities as pq
from exana.tracking.fields import (gridness, occupancy_map,
                     spatial_rate_map,
                     spatial_rate_map_1d)
from exana.tracking.head import *
from utils import simpleaxis
import math
from scipy.ndimage.measurements import center_of_mass
import matplotlib.gridspec as gridspec
from exana.misc.tools import is_quantities

def plot_path(x, y, t, box_xlen, box_ylen, sptr=None,
              color='grey', alpha=0.5, origin='upper',
              spike_color='r', rate_markersize=False, markersize=10.,
              animate=False, ax=None, title=''):
    """
    Plot path visited

    Parameters
    ----------
    x : quantities.Quantity array in m
        1d vector of x positions
    y : quantities.Quantity array in m
        1d vector of y positions
    t : quantities.Quantity array in s
        1d vector of time at x, y positions
    sptr : neo.SpikeTrain
    box_xlen : quantities scalar
        size of spatial 2d square
    box_ylen : quantities scalar
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
    is_quantities([box_xlen, box_ylen], 'scalar')
    is_quantities([x, y, t], 'vector')
    box_xlen = float(box_xlen.rescale('m').magnitude)
    box_ylen = float(box_ylen.rescale('m').magnitude)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, xlim=[0, box_xlen], ylim=[0, box_ylen],
                             aspect=1)
    if sptr is not None:
        spikes_in_bin, _ = np.histogram(sptr, t)
        is_spikes_in_bin = np.array(spikes_in_bin, dtype=bool)

        if rate_markersize:
            markersizes = spikes_in_bin[is_spikes_in_bin]*markersize
        else:
            markersizes = markersize*np.ones(is_spikes_in_bin.size)
    if animate:
        import time
        plt.show()
        for idx, x, y, active, msize in zip(range(len(x)), x, y):
            ax.plot(x, y, c=color, alpha=alpha)
            if sptr is not None:
                if is_spikes_in_bin[idx]:
                    ax.scatter(x, y, facecolor=spike_color, edgecolor=spike_color,
                               s=markersizes[idx])
            time.sleep(0.1)  # plt.pause(0.0001)
            plt.draw()
    else:
        ax.plot(x, y, c=color, alpha=alpha)
        if sptr is not None:
            ax.scatter(x[0:-1][is_spikes_in_bin], y[0:-1][is_spikes_in_bin],
                       facecolor=spike_color, edgecolor=spike_color,
                       s=markersizes)
    ax.set_title(title)
    ax.grid(False)
    if origin == 'upper':
        ax.invert_yaxis()
    return ax


def plot_head_direction_rate(sptr, ang_bins, rate_in_ang, projection='polar',
                             normalization=False, ax=None, color='k'):
    """


    Parameters
    ----------
    sptr : neo.SpikeTrain
    ang_bins : angular binsize
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
    assert ang_bins.units == pq.degrees, 'ang_bins must be in degrees'
    if normalization:
        rate_in_ang = normalize(rate_in_ang, mode='minmax')
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=projection)
    binsize = ang_bins[1] - ang_bins[0]
    if projection is None:
        ax.set_xticks(range(0, 360 + 60, 60))
        ax.set_xlim(0, 360)
    elif projection == 'polar':
        ang_bins = [math.radians(deg) for deg in ang_bins] * pq.radians
        binsize = math.radians(binsize) * pq.radians
        ax.set_xticks([0, np.pi])
    ax.bar(ang_bins, rate_in_ang, width=binsize, color=color)
    return ax


def plot_ratemap(x, y, t, sptr, binsize=0.05*pq.m, box_xlen=1*pq.m,
                 box_ylen=1*pq.m, vmin=0, ax=None, mask_unvisited=True, convolve=True,
                 origin='upper', cmap='jet'):
    """


    Parameters
    ----------
    x : 1d vector of x positions
    y : 1d vector of y positions
    t : 1d vector of time at x, y positions
    sptr : one neo.SpikeTrain
    binsize : size of spatial 2d square bins
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

    rate_map = spatial_rate_map(x, y, t, sptr, binsize=binsize,
                                 mask_unvisited=mask_unvisited,
                                 box_xlen=box_xlen, box_ylen=box_ylen,
                                 convolve=convolve)
    ax.imshow(rate_map, interpolation='none', origin=origin,
              extent=(0, 1, 0, 1), vmin=vmin, cmap=cmap)
    ax.set_title('%.2f Hz' % np.nanmax(rate_map))
    ax.grid(False)
    return ax


def plot_ratemap_linear_track(x, t, sptr,
                              binsize=0.05*pq.m,
                              track_len=2*pq.m,
                              end_0=[],
                              end_1=[],
                              vmin=0,
                              ax=None,
                              mask_unvisited=True,
                              convolve=True,
                              origin='upper',
                              cmap='jet'):
    """
    Plot ratemap along one dimension

    Parameters
    ----------
    x : 1d vector of x positions
    t : 1d vector of time at x
    sptr : one neo.SpikeTrain
    binsize : size of spatial bins
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
    rate_map = spatial_rate_map_1d(x, t, sptr, binsize=binsize,
                                   mask_unvisited=mask_unvisited,
                                   box_size=box_size,
                                   convolve=convolve)
    ax.imshow(rate_map, interpolation='none', origin=origin,
              extent=(0, 1, 0, 1), vmin=vmin, cmap=cmap)
    ax.grid(False)
    return ax


def plot_ratemaps_linear_track(x, t, sptrs,
                               binsize=0.05*pq.m,
                               track_len=2*pq.m,
                               end_0=[],
                               end_1=[],
                               vmin=0,
                               vmax=None,
                               mask_unvisited=True,
                               convolve=True,
                               origin='upper',
                               cmap='jet',
                               return_bins=False):
    """
    Plot ratemaps along one dimension for multiple neurons

    Parameters
    ----------
    x : 1d vector of x positions
    t : 1d vector of time at x
    sptr : one neo.SpikeTrain
    binsize : size of spatial bins
    vmin : color min
    ax : matplotlib axes
    mask_unvisited : True: mask bins which has not been visited

    Returns
    -------
    out : axes

    """

    n_sptr = len(sptrs)

    rate_maps = []
    coms = []  # centers of masses
    unit_names = []
    for sptr in sptrs:
        # compute rate maps
        rate_map, bins = spatial_rate_map_1d(x, t, sptr,
                                             binsize=binsize,
                                             track_len=track_len,
                                             mask_unvisited=True,
                                             convolve=False,
                                             return_bins=True,
                                             smoothing=0.02)
        rate_maps.append(rate_map)
        # calc center of mass
        com = center_of_mass(rate_map)[0]
        coms.append(com)
        unit_names.append(sptr.unit.name)
    # set vmax to max rate if it does not exist
    if vmax is None:
        vmax = np.max(np.array(rate_maps).flatten())
    com_ordering = np.argsort(coms)

    fig = plt.figure(figsize=(1*n_sptr, 8))
    fig.suptitle('Spatial rate maps on linear track')
    gs = gridspec.GridSpec(n_sptr, 1)
    gs.update(hspace=0.0)

    for i, i_ord in enumerate(com_ordering):
        ax = fig.add_subplot(gs[i])
        map_i = np.expand_dims(rate_maps[i_ord], axis=0)
        im = ax.imshow(map_i,
                       interpolation='none',
                       origin=origin,
                       vmin=vmin,
                       vmax=vmax,
                       extent=(0, track_len.rescale('m').magnitude,
                               0, 1/n_sptr*2),
                       cmap=cmap)
        ax.set_ylabel(str(unit_names[i_ord]))
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        if i+1 < len(com_ordering):
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.spines['bottom'].set_visible(False)
        if i+1 == len(com_ordering):
            ax.set_xticklabels
            ax.set_xlabel('x [' + str(x.units) + ']')

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    if return_bins:
        return fig, bins
    else:
        return fig
    # Todo: Empty bin on left


def plot_occupancy(x, y, t, binsize=0.05*pq.m, box_xlen=1*pq.m, box_ylen=1*pq.m,
                  vmin=0, ax=None, convolve=True,
                  origin='upper', cmap='jet'):
    """


    Parameters
    ----------
    x : 1d vector of x positions
    y : 1d vector of y positions
    t : 1d vector of time at x, y positions
    sptr : one neo.SpikeTrain
    binsize : size of spatial 2d square bins
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

    occ_map = occupancy_map(x, y, t, binsize=binsize, box_xlen=box_xlen,
                             box_ylen=box_ylen, convolve=convolve)
    cax = ax.imshow(occ_map, interpolation='none', origin=origin,
                   extent=(0, 1, 0, 1), vmin=vmin, cmap=cmap, aspect='auto')
    # ax.set_title('%.2f s' % np.nanmax(occ_map))
    ax.grid(False)
    return cax, np.nanmax(occ_map)
