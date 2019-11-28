import numpy as np


def _inf_rate(rate_map, px):
    '''
    A helper function for information rate.

    Originally from https://github.com/MattNolanLab/gridcells
    '''
    tmp_rate_map = rate_map.copy()
    tmp_rate_map[np.isnan(tmp_rate_map)] = 0
    avg_rate = np.sum(np.ravel(tmp_rate_map * px))
    return (np.nansum(np.ravel(tmp_rate_map * np.log2(tmp_rate_map/avg_rate) *
            px)), avg_rate)


def sparsity(rate_map, px):
    '''
    Compute sparsity of a rate map, The sparsity  measure is an adaptation
    to space. The adaptation measures the fraction of the environment  in which
    a cell is  active. A sparsity of, 0.1 means that the place field of the
    cell occupies 1/10 of the area the subject traverses [2]_

    Parameters
    ----------
    rate_map : numpy.ndarray
        A firing rate map, any number of dimensions.

    Returns
    -------
    out : float
        sparsity

    References
    ----------
    .. [2] Skaggs, W. E., McNaughton, B. L., Wilson, M., & Barnes, C. (1996).
       Theta phase precession in hippocampal neuronal populations and the
       compression of temporal sequences. Hippocampus, 6, 149-172.
    '''
    tmp_rate_map = rate_map.copy()
    tmp_rate_map[np.isnan(tmp_rate_map)] = 0
    avg_rate = np.sum(np.ravel(tmp_rate_map * px))
    avg_sqr_rate = np.sum(np.ravel(tmp_rate_map**2 * px))
    return avg_rate**2 / avg_sqr_rate


def selectivity(rate_map, px):
    '''
    "The selectivity measure max(rate)/mean(rate)  of the cell. The more
    tightly concentrated  the cell's activity, the higher the selectivity.
    A cell with no spatial tuning at all will  have a  selectivity of 1" [2]_.

    Parameters
    ----------
    rate_map : numpy.ndarray
        A firing rate map, any number of dimensions.

    Returns
    -------
    out : float
        selectivity
    '''
    tmp_rate_map = rate_map.copy()
    tmp_rate_map[np.isnan(tmp_rate_map)] = 0
    avg_rate = np.sum(np.ravel(tmp_rate_map * px))
    max_rate = np.max(np.ravel(tmp_rate_map))
    return max_rate / avg_rate


def information_rate(rate_map, px):
    '''
    Compute information rate of a cell given variable x.
    A simple algorithm devised by [1]_. This computes the spatial information
    rate of cell spikes given variable x (e.g. position, head direction) in
    bits/second. This function is copied from Lucas Solanka, Matt Nolans lab

    Parameters
    ----------
    rate_map : numpy.ndarray
        A firing rate map, any number of dimensions. If units are in Hz, then
        the information rate will be in bits/s.
    px : numpy.ndarray
        Probability density function for variable ``x``. ``px.shape`` must be
        equal ``rate_maps.shape``

    Returns
    -------
    I : float
        Information rate.

    Notes
    -----
    Quote from [1]_:
    "To get the basic idea, imagine we are recording the activity of a neuron
    in the brain of a rat, while the rat is wandering around randomly on a
    circular platform. Suppose we observe that the cell fires only when the
    rat is on the left half of the platform, and that it fires at a constant
    rate everywhere on the left half; and suppose that on the whole the rat
    spends half of its time on the left half of the platform. In this case, if
    we are prevented from seeing where the rat is, but are informed that the
    neuron has just this very moment fired a spike, we obtain thereby one bit
    of information about the current location of the rat. Suppose we have a
    second cell, which fires only in the southwest quarter of the platform; in
    this case a spike would give us two bits of information. If there were in
    addition a small amount of background firing, the information would be
    slightly less than two bits. And so on.". Invalid positions are masked with
    ``np.nan`` and replaced with 0.

    References
    ----------
    .. [1] Skaggs, W.E. et al., 1993. An Information-Theoretic Approach to
       Deciphering the Hippocampal Code. In Advances in Neural Information
       Processing Systems 5. pp. 1030-1037.
    '''
    return _inf_rate(rate_map, px)[0]


def information_specificity(rate_map, px):
    '''
    Compute the 'specificity' of the cell firing rate to a variable X.
    Compute :func:`information_rate` for this cell and divide by the average
    firing rate of the cell. See [1]_ for more information.

    Originally from https://github.com/MattNolanLab/gridcells

    Parameters
    ----------
    rate_map : numpy.ndarray
        A firing rate map, any number of dimensions.
    px : numpy.ndarray
        Probability density function for variable ``x``. ``px.shape`` must be
        equal ``rate_maps.shape``

    Returns
    -------
    I : float
        Information in bits/spike.
    '''
    I, avg_rate = _inf_rate(rate_map, px)
    return I / avg_rate


def prob_dist(x, y, bins):
    '''
    Calculate a probability distribution for animal positions in an arena.

    Parameters
    ----------
    x : quantities.Quantity array in m
    y : quantities.Quantity array in m
    bins : quantities.Quantity array in m

    Returns
    -------
    dist : numpy.ndarray
        Probability distribution for the positional data. The first dimension
        is the y axis, the second dimension is the x axis.
    '''

    H, _, _ = np.histogram2d(x, y, bins=bins, normed=False)
    return (H / len(x))


def prob_dist_1d(x, bins):
    '''
    Calculate a probability distribution for animal positions in an arena.

    Parameters
    ----------
    x : quantities.Quantity array in m
    bins : quantities.Quantity array in m

    Returns
    -------
    dist : numpy.ndarray
        Probability distribution for the positional data. The first dimension
        is the y axis, the second dimension is the x axis.
    '''

    H, _ = np.histogram(x, bins=bins, normed=False)
    return (H / len(x)).T


def population_vector_correlation(rmaps1, rmaps2,
                                  mask_nans=False,
                                  return_corr_coeff_map=False):
    """
    Calcualte population vector correlation between two
    stacks of rate maps.

    Parameters
    ----------
    rmaps1 : ndarray
    Array of the shape [n_units, n_bins_dim1, n_bins_dim2]
    rmaps2 : ndarray
    Array of the shape [n_units, n_bins_dim1, n_bins_dim2]
    mask_nans : bool
    If mask_nans, nan-values will be excluded for x-, y-bin
    from the correlation calculation
    return_corrcoeffs : bool
    If return_corr_coeff_map, return the correlation coefficient
    map instead for the mean.

    Returns
    -------
    pop_vec_corr : float
    Population vector correlation
    """
    bins_x = rmaps1.shape[1]
    bins_y = rmaps1.shape[2]

    assert rmaps1.shape == rmaps2.shape
    # correlation coefficient requires at least 2x2 values
    assert rmaps1.shape[0] > 1

    corr_coeff_map = np.zeros((bins_x, bins_y))
    corr_coeff_map[:] = np.nan

    for i in range(bins_x):
        for j in range(bins_y):
            xy1 = rmaps1[:, i, j]
            xy2 = rmaps2[:, i, j]
            if mask_nans:
                bool_nan_xy1 = np.isnan(xy1)
                bool_nan_xy2 = np.isnan(xy2)

                mask_invalid = np.logical_or(
                    bool_nan_xy1,
                    bool_nan_xy2)
                mask_valid = ~mask_invalid
                xy1 = xy1[mask_valid]
                xy2 = xy2[mask_valid]

            corr_coeff_map[i, j] = np.corrcoef(
                xy1,
                xy2)[0, 1]

    pop_vec_corr = np.nanmean(corr_coeff_map)

    if return_corr_coeff_map:
        return pop_vec_corr, corr_coeff_map
    else:
        return pop_vec_corr
