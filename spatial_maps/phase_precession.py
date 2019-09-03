import numpy as np


class PassMask(object):
    """holds masks between t_enter and t_exit for the chosen kinds of data. """
    def __init__(self, t_enter, t_exit, data, data_names=None):
        """t_enter, t_exit : time of entering and exiting.
        
        data : list of array likes
            time_series to store masks for. 

        data_names : list of strings (optional)
            allows dict-based indexing of data-sets if supplied. """
        self._t_enter = t_enter
        self._t_exit = t_exit


        if data_names is None:
            self.data_masks = [np.logical_and(t>t_enter, t<=t_exit) \
                               for t in data]
        else:
            if len(data_names) != len(data):
                raise ValueError('data_names and data must have same length')
            self.data_masks = {name:np.logical_and(t>t_enter, t<=t_exit) \
                               for t, name in zip(data, data_names)}

    def __getitem__(self, index):
        return self.data_masks[index]
