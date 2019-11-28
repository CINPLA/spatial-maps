import numpy as np


def border_score(rate_map, fields):
    """
    Uses a separation of the fields in a rate map to calculate a border
    score as described in [1].
    Parameters
    ----------
    rate_map : np 2d array
        firing rate in each bin
    fields : np 2d array of ints
        areas all filled with same value, corresponding to fields
        in rate_map. See output of separate_fields
    References
    ----------
    [1]: Geoffrey W. Diehl, Olivia J. Hon, Stefan Leutgeb, Jill K. Leutgeb,
    https://doi.org/10.1016/j.neuron.2017.03.004
    :Authors:
        Halvard Sutterud <halvard.sutterud@gmail.com>
    """
    from scipy.ndimage import labeled_comprehension

    # Find parts of fields next to border. Use a second to outer bins, as
    # outer bins won't be specified as field
    if np.all(fields == 0):
        raise ValueError("Must have at least one field")

    inner = np.zeros(np.array(rate_map.shape)-(4,4),dtype=bool)
    wall = np.pad(inner, 1, 'constant', constant_values=[[0,0],[1,0]])
    wall = np.pad(wall, 1, 'constant', constant_values=0)
    max_extent = 0

    ones = np.ones_like(rate_map)

    for i in range(4):
        borders = np.logical_and(fields > 0, wall)
        extents = labeled_comprehension(
            input=borders, labels=fields, index=None, func=np.sum,
            out_dtype=np.int64, default=0)
        max_extent = np.max([max_extent, np.max(extents)])

        # dont rotate the fourth time
        wall = np.rot90(wall) if i < 3 else wall

    C_M = max_extent / rate_map.shape[0]

    x = np.linspace(-0.5, 0.5, rate_map.shape[1])
    y = np.linspace(-0.5, 0.5, rate_map.shape[0])
    X,Y = np.meshgrid(x,y)

    # create linear increasing value towards middle
    dist_to_nearest_wall = 1 - (np.abs(X + Y) + np.abs(X - Y))

    d_m = np.average(dist_to_nearest_wall, weights=rate_map)
    b = (C_M - d_m) / (C_M + d_m)
    return b
