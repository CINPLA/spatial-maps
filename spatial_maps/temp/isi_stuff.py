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
    # print(passes)


def simulate_spiketrain(t, rate, modulation_func = np.ones_like,ax=None):
    """Create spiketrain modulated by modulation_func, normalized with rate"""
    import elephant.spike_train_generation as stg
    import elephant.statistics as ep_stats
    sampling_rate = 1/(np.mean(np.diff(t)))
    sim_rate = modulation_func(t.magnitude)

    sim_rate /= np.sum(sim_rate)
    dT = t[-1] - t[0]
    sim_rate *= dT*rate.magnitude*sampling_rate.magnitude

    one_sec = np.argmin(np.abs(t-1*pq.s))
    if ax is not None:
        ax.plot(t[:one_sec],sim_rate[:one_sec])


    rate_array = neo.AnalogSignal(sim_rate, units = 'Hz', sampling_rate = sampling_rate, t_start=t[0])
    sim_sptr = stg.inhomogeneous_poisson_process(rate_array)

    print(len(sim_sptr))
    isi = ep_stats.isi(sim_sptr)

    return sim_sptr, isi

def exp_cumsum(x,l):
    return 1-np.exp(-l*x)


# def save_LFP_theta_timefrequency(exdir_path, out_path = None, date = None,
#         z_normalized = True, downsample = 5, freq_range =  [4*pq.Hz, 20*pq.Hz],
#         deltafreq = 0.2*pq.Hz, f0 = 8):
#     """Calculates the timefrequency map with a wavelet transform for the 250 Hz LFP signals in
#     exdir_path in the theta range. These are then saved locally to a exdir file, with the
#     structure 
#     main.exdir/analysis/timefrequency/
#         channelX1/
#             frequencies/
#             map/
#             times/
#         channelX2/
#             frequencies/
#             map/
#             times/
        
#     Parameters
#     ----------
#     exdir_path : string
    
#     out_path : string (optional)
    
#     date : string (optional)
#         if out_path is None, the date is used as a subfolder of a very
#         specific file location on a specific computer (this computer). If
#         not specified, the current date is used with format %Y-%m-%d.

#     z_normalized : default True
#         whether to z-normalize signal before wavelet transform
#     downsample : int, default 5
#         downsample signal to reduce redundancy in map size. 1 gives no
#         downsampling

#     freq_range : default  [4*pq.Hz, 20*pq.Hz]
#         frequency range for wavelet transform
#     deltafreq : default 0.2*pq.Hz
#         frequency spacing for wavelet transform
#     f0 : default 8
#         parameter for wl

#     """
#     import exdir
#     import exdir.plugins.quantities
#     from . import tools
#     import datetime
#     import os

#     session_id = exdir_path.split('/')[-2]
#     anas = tools.get_lfp_data(exdir_path, local=False, avoid_duplicates = True)

#     import quantities as pq
#     from exana.time_frequency import timefreq

#     anas = [ana for ana in anas if ana.sampling_rate == 250*pq.Hz]
#     # Check that there is no signals with same channel

#     group_ids = [ana.annotations['electrode_group_id'] for ana in anas]
#     assert len(group_ids) == len(np.unique(group_ids))


#     date = date or datetime.datetime.now().strftime('%Y-%m-%d')
#     if out_path is None:
#         out_path = '/home/halvard/cinpla/cinpla-work/analysis/' + date + '/'
#     if not os.path.exists(out_path):
#         os.mkdir(out_path)

#     out_file = exdir.File(out_path+session_id, plugins= [exdir.plugins.quantities])
#     group = out_file.require_group('analysis/timefrequency')
    

    
#     for ana in anas:
#         sample_rate = 250*pq.Hz
#         if z_normalized:
#             ana = (ana- np.mean(ana))/np.std(ana)
            
#         new_ana = ana[::downsample]
#         new_ana.sampling_rate = ana.sampling_rate/downsample
#         f_start = freq_range[0]
#         f_stop = freq_range[1]

#         tf = timefreq.TimeFreq(new_ana, f_start = f_start, f_stop = f_stop, 
#                        f0 = f0, deltafreq = deltafreq, 
#                        sampling_rate = new_ana.sampling_rate,
#                        optimize_fft=True)

#         channel_group = group.require_group('channel_group{}'.format(ana.annotations['electrode_group_id']))
#         channel_group.description = 'Wavelet analysis of theta frequencies'
#         channel_group.attrs = { 'z_normalized':z_normalized, 'sample_rate' : sample_rate }

#         channel_group.require_dataset('frequencies', data = tf.freqs)
#         channel_group.require_dataset('map', data = tf.map)
#         channel_group.require_dataset('times', data = tf.times)

from exana.time_frequency import timefreq 

class TimeFreqClone(timefreq.TimeFreq):
    """Just for storing the parameters. """

    def __init__(self, im_map, freqs, times):
        """TODO: to be defined1. """
        self.map = im_map
        self.freqs = freqs
        self.times = times


        

def load_LFP_theta_timefrequency(exdir_path, date = None, in_path = None):
    import exdir
    import exdir.plugins.quantities

    if not (date or in_path):
        raise  TypeError ('Cannot load LFP spectrum. Missing date or path keyword.')
    elif date:
        if type(date) == 'datetime.datetime':
            date = date.strftime('%Y-%m-%d')
        in_path = '/home/halvard/cinpla/cinpla-work/analysis/' + date + '/'
    else:
        pass

    session_id = exdir_path.split('/')[-2]
    in_file = exdir.File(in_path+session_id, 'r', plugins= [exdir.plugins.quantities])
    group = in_file.require_group('analysis/timefrequency')
    
    data = {}

    for group_name, channel_group in group.items():
        print(group_name)
        wl_map = np.array(channel_group['map'].data)
        freq   = np.array(channel_group['frequencies'].data)
        times  = np.array(channel_group['times'].data)
        data[group_name] = TimeFreqClone(wl_map, freq,times)

    in_file.close()

    return data
        

# def save_LFP_power_spectrum(exdir_path,out_path = None, date = None,
#         z_normalized = True):
#     """Calculates the power spectrum for the 250 Hz LFP signals in
#     exdir_path. These are then saved locally to a exdir file, with the
#     structure 
#     main.exdir/analysis/LFP_spectrum/
#         frequencies/
#         spectrum_1/
#         spectrum_2/
        
#     Parameters
#     ----------
#     exdir_path : string
    
#     out_path : string (optional)
    
#     date : string (optional)
#         if out_path is None, the date is used as a subfolder of a very
#         specific file location on a specific computer (this computer). If
#         not specified, the current date is used with format %Y-%m-%d.
#     """
#     from . import tools
#     import datetime
#     import os

#     session_id = exdir_path.split('/')[-2]
#     anas = tools.get_lfp_data(exdir_path, local=False)

#     import quantities as pq

#     sample_rate = 250*pq.Hz
        
#     fpre, Pxxpre = tools.get_LFP_spectrum(anas, sample_rate = sample_rate,
#             z_normalized = z_normalized)
    
#     date = date or datetime.datetime.now().strftime('%Y-%m-%d')

#     if out_path is None:
#         out_path = '/home/halvard/cinpla/cinpla-work/analysis/' + date + '/'
#         # import glob
#         # n = len(glob.glob(out_path + '*')) 
#         # out_path +=  '_{}/'.format(n)

#     if not os.path.exists(out_path):
#         os.mkdir(out_path)
        
#     import exdir
#     import exdir.plugins.quantities
    
#     out_file = exdir.File(out_path+session_id, plugins= [exdir.plugins.quantities])
#     group = out_file.require_group('analysis/LFP_spectrum')
    
#     group.description = 'Approximations of LFP spectrums, using tools.get_LFP_signal'
#     group.attrs = {'old': False,
#                     'z_normalized':z_normalized,
#                     'nchunks'     : 1,
#                    'chunksize'   : 'everything',
#                    'sample_rate' : sample_rate }

#     fpre_data = group.require_dataset('frequencies', data = fpre)
#     pxx1 = group.require_dataset('spectrum_1', data = Pxxpre[0])
#     pxx2 = group.require_dataset('spectrum_2', data = Pxxpre[1])

# def load_LFP_power_spectrum(exdir_path, date = None, in_path = None):
#     import exdir
#     import exdir.plugins.quantities

#     if not (date or in_path):
#         raise  TypeError ('Cannot load LFP spectrum. Missing date or path keyword.')
#     elif date:
#         if type(date) == 'datetime.datetime':
#             date = date.strftime('%Y-%m-%d')
#         in_path = '/home/halvard/cinpla/cinpla-work/analysis/' + date + '/'

#     else:
#         in_path = path

#     session_id = exdir_path.split('/')[-2]
#     in_file = exdir.File(in_path+session_id, 'r', plugins= [exdir.plugins.quantities])
#     group = in_file.require_group('analysis/LFP_spectrum')
    
#     freq_data = group['frequencies'] # require_dataset('frequencies')
#     pxx1_data = group['spectrum_1'] # require_dataset('spectrum_1')
#     pxx2_data = group['spectrum_2'] # require_dataset('spectrum_2')
#     pxx1 = np.array(pxx1_data.data)
#     pxx2 = np.array(pxx2_data.data)
#     freq = np.array(freq_data.data)
#     in_file.close()

#     return freq, pxx1, pxx2

def isi_passes(exdir_path, return_slow_blocks = False, min_duration=4*pq.s,
        binsize = 0.02*pq.m, smoothing = 0.025):
    """Loads data set and calculates inter spike interval from passes. May
    also return isi for blocks where speed is low, as described in [1]

    Parameters
    ----------
    exdir_path : string

    Returns:
    --------
        passes : pandas DataFrame 
            two columns, field and isi for each pass 

        if return_slow_blocks:
        slow_blocks : list
            isi of each block of at least min_duration

    References
    ----------

    [1] : Burak, Y., & Fiete, I. R. (2009). PLoS computational biology, 5(2), e1000291.

    """

    import exana.tracking as tr

    x,y,t,speed,sptr = tools.load_data(exdir_path)
    rate_map =  tr.spatial_rate_map(x, y, t, sptr, binsize=binsize, 
                                smoothing = smoothing,
                           mask_unvisited=False, convolve = True)

    avg_dist = tr.find_avg_dist(rate_map, thrsh = 0)

    rm_sep =  tr.spatial_rate_map(x, y, t, sptr, binsize=0.01*pq.m, 
                                smoothing = 0.2*avg_dist, convolve_spikes_and_time=True,
                           mask_unvisited=False, convolve = True)
    f,nf,bc = tr.separate_fields(rm_sep, laplace_thrsh=0.02/avg_dist)

    from work.analyse_field_traversal import field_traversals
    df = pd.DataFrame(field_traversals(x,y,t, f.T, index = np.arange(np.max(f)+1)))

    # pooled = []
    # ind_pooled = []
    # temp_mean = []
    # temp_std = []

    isis = []
    field_indices = []

    dt = np.median(np.diff(t))

    for index,data in df.T.items():
        spikes = sptr[np.logical_and(sptr >= data.t[0], sptr < data.t[-1]+dt)]
        n = len(spikes)/len(data.t)
        
        # ISI = np.diff(spikes.times)
        if data.field > 0:
            isis.append(np.diff(spikes.times))
            field_indices.append(data.field)

    # df = pd.DataFrame()
    # df['isi'] = isis
    # df['fields'] = field_indices
    df = [isis, field_indices]


    if return_slow_blocks:
        from astropy.convolution import Gaussian1DKernel, convolve_fft
        kernel = Gaussian1DKernel(70)
        speed = convolve_fft(speed, kernel)

        slow_blocks = []
        mask = speed < 0.08 *pq.m/pq.s

        from scipy import ndimage
        lbl, nlbl = ndimage.label(mask)
        obj = ndimage.find_objects(lbl)

        for o in obj:
            times = t[o]
            duration = times[-1] - times[0]
            if duration < min_duration:
                continue
            spikes = sptr[np.logical_and(sptr > times[0], 
                                         sptr < times[-1]+dt)]

            if spikes.size > 2:
                slow_blocks.append(np.diff(spikes.times))

        return df, slow_blocks 
    else:
        return df 
