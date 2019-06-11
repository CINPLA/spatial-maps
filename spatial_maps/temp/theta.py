"""
File: analysis_scripts.py
Author: Halvard Sutterud
Email: halvard.sutterud@gmail.com
Github: https://github.com/halvarsu
Description: Collection of higher abstracted analysis scripts, often
depending on several other scripts.
"""

import numpy as np
import matplotlib.pyplot as plt
from . import tools
from . import analyse_field_traversal as aft
import pandas as pd
import quantities as pq

        
def theta_spike_model_compare(x, y, t, sptr, f, bc, exdir_path, n_samples = 5):
    """Extracts spikes of passes through the fields f, and compares with a
    theta-modulated poisson-process with the same rate, normalized such that the
    mean rate equals the inverse mean interspike of the pass.

    Parameters
    ==========
    :x,y,t: numpy arrays
        positions and times for recording
    :sptr: neo analogsignal
        spike times
    :f: NxN numpy array
        spiking fields, regions of higher spiking where the spike
        characteristics is to be studied.
    :bc: 2xNf numpy array
        x and y positions of bump centers
    :exdir_path:
        path to data 
    :n_samples: int
        number of samples of the poisson process per run.

    Returns
    =======
    :cv_data: pd DataFrame 
        cv and theta data of each spiketrain with > 2 spikes in passes 
    :sim_cv_data: pd DataFrame 
        cv and theta data of each simulated spiketrain with > 2 spikes 
    """

    from scipy.interpolate import interp1d
    import quantities as pq

    signal, filt_sig, phase = tools.get_signals(exdir_path)
    phase_func = interp1d(signal.times, phase, 
            fill_value = (phase[0] , phase[-1]), bounds_error = False)
    sig_func = interp1d(signal.times, filt_sig,
            fill_value = (filt_sig[0] , filt_sig[-1]), bounds_error = False)
    lowres_sig = sig_func(t)

    passes = aft.get_pass_data(x, y, t, sptr, f, bc*pq.m, phase_func =
            phase_func, dist_func_method='scikit')
    pass_data = pd.DataFrame(passes)

    pass_data = aft.simulate_pass_sptr_with_theta(pass_data, t, lowres_sig,
            n_samples = n_samples)

    cv_data = []
    sim_cv_data = []

    for ind,p in pass_data.T.items():
        pass_sptr = p['spikes']
        sim_spiketrains = p['sim_sptr']
        
        if len(pass_sptr)>2:#not np.isnan(cv):
            cv,std,mean,isi = tools.isi_statistics(pass_sptr)
            cv_data.append({ 'cv':cv,'std':std,'mean':mean ,'isi':isi})
            
        for sim_sptr in sim_spiketrains:
            if len(sim_sptr) > 2:
                sim_cv,sim_std,sim_mean,sim_isi = tools.isi_statistics(sim_sptr)
                sim_cv_data.append({'cv':sim_cv,'std':sim_std,'mean':sim_mean,'isi':sim_isi})
            
    cv_data = pd.DataFrame(cv_data)
    sim_cv_data = pd.DataFrame(sim_cv_data)

    return cv_data, sim_cv_data
