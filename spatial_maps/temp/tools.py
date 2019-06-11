"""
File: tools.py
Author: Halvard Sutterud
Email: halvard.sutterud@gmail.com
Github: https://github.com/halvarsu
Description: Lots of small functions for loading/writing/manipulating/analysing data.
"""

print("tools.py: THIS IS PROBABLY DEPRECATED. Mikkel (eller halvard) se på dette")

import exdir 
# import exana.tracking as tr
import spatialspikefields as ssf
import expipe
import numpy as np
import os
import neo
import quantities as pq
import copy

def get_pos_data(exdir_path, par = {'pos_fs': 100*pq.Hz, 'f_cut':6*pq.Hz}, 
                 exdir_group=None,return_group=False, simple_structure=False):
    """Loads position data from exdir, either from an axona file structure or
    for a more simple customized structure. Can use already opened groups
    to reduce overhead."""
    
    if exdir_group is None:
        exdir_group = exdir.File(exdir_path)
    if simple_structure:
        position_group = exdir_group['position_data']
        pgx = position_group['x']
        pgy = position_group['y']
        pgt = position_group['t']
        x = pq.Quantity(pgx.data, pgx.attrs['unit'])
        y = pq.Quantity(pgy.data, pgy.attrs['unit'])
        t = pq.Quantity(pgt.data, pgt.attrs['unit'])
    else:
        position_group = exdir_group['processing']['tracking']['camera_0']['Position']
        stop_time = position_group.attrs.to_dict()["stop_time"]
        x1, y1, t1 = tr.get_raw_position(position_group['led_0'])
        x2, y2, t2 = tr.get_raw_position(position_group['led_1'])
        x, y, t = tr.select_best_position(x1, y1, t1, x2, y2, t2)
        x, y, t = tr.interp_filt_position(x, y, t, pos_fs=par['pos_fs'], f_cut=par['f_cut'])
        mask = t <= stop_time
        #mask = t2 <= stop_time
        x = x[mask]
        y = y[mask]
        t = t[mask]
    dt = np.mean(np.diff(t))
    vel = np.gradient([x,y],axis=1)/dt
    speed = np.linalg.norm(vel,axis=0)

    if return_group:
        return x,y,t,speed, exdir_group
    else:
        return x,y,t,speed

def get_signals(exdir_path):
    """Supplies all your LFP needs, giving you perfect combination of
    signal and functions.
     """
    analog_signals = get_lfp_data(exdir_path, local = False)

    sptr_ch_group = int(exdir_path.split('/')[5][-1])
    signal = pick_correct_lfp(analog_signals, sptr_ch_group)
    filt_sig, _, phase = process_signal(signal)
    return signal, filt_sig, phase


def get_lfp_data(exdir_path, exdir_group=None, 
        data_path = expipe.settings['data_path'], local = True,
        avoid_duplicates = True):
    analogsignals = []
    if local:
        data_path = '/home/halvard/cinpla/cinpla-work/exdir_data/'
        file_path =  data_path + "/".join(exdir_path.split('/')[1:3])
    else:
        file_path = data_path + "/".join(exdir_path.split('/')[:3])

    if exdir_group is None:
        import exdir.plugins.quantities
        exdir_group = exdir.File(file_path, 
                plugins = [exdir.plugins.quantities], mode = 'r')

    for channel_group in exdir_group['processing']['electrophysiology'].values():
        try:
            for lfp_group in channel_group['LFP'].values():
                print(lfp_group.name)
                try:
                    ana = read_LFP_from_exdir(lfp_group,  exdir_path, # lfp_group.name,
                                                 cascade=True, lazy=False)
                    if (avoid_duplicates and 
                            ana.annotations in [other.annotations for other in analogsignals]):
                        pass
                    else:
                        analogsignals.append(ana)

                except KeyError as err:
                    print(lfp_group.name,'gave error:', err)
                    pass
        except KeyError as err:
            pass
    return analogsignals


def pick_correct_lfp(analog_signals, sptr_ch_group, low_res = True):
    """
    assumes channel groups 0-3 and 4-7 are separate groups, and finds the
    correct signal for a spiketrain

    Parameters:
    analog_signals : list of neo AnalogSignals
    sptr_ch_group : int, channel group to match hemisphere with
    low_res : bool, pick low res or high_res signal
    
    returns:
    """
    rates = np.array([s.annotations['sample_rate'] for s in analog_signals])
    is_low_res = rates <= 1000*pq.Hz
    should_be_low_res = low_res

    signals = [s for i,s in enumerate(analog_signals) if (is_low_res[i] == should_be_low_res)]
    
    sptr_hemisphere = sptr_ch_group > 3
    sig_hemispheres = [s.annotations['electrode_group_id'] > 3 for s in signals]
    i = sig_hemispheres.index(sptr_hemisphere)
    sig = signals[i]
    
    print(f'picking channel from {list(range(4*i,4*(i+1)))}')
    print(f'sptr channel: {sptr_ch_group}, signal channel: {sig.annotations["electrode_group_id"]}')
    return sig


def process_signal(s, pad = True):
    """Filters signal, performs hilbert transform and extracts phase.
    Parameters:
    -----------
    s : neo AnalogSignal
        signal 

    pad : bool
        pad signal before hilbert transform

    Returns 
    -------
    filt : neo AnalogSignal
        filtered signal, same attributes and annotations as input otherwise
    analytic_signal : np array
        hilbert transformed signal
    filt : np array
        phase
    """
    

    from scipy.fftpack import next_fast_len
    from exana.misc.signal_tools import filter_analog_signals
    from scipy.signal import hilbert

    fs = s.sampling_rate
    filt_signal = filter_analog_signals(s.T[0], [7,12]*pq.Hz , fs)

    # pad with zeros for the hilbert transform to speed up several orders of magnitude
    n= len(filt_signal)
    N = next_fast_len(n)

    if pad:
        print('length %d, next fast length %d: padding with %d zeros' %(n,N,N-n))
        c = np.pad(filt_signal,(0,N-n),'constant')
        analytic_signal = hilbert(c)[:n]
    else:
        analytic_signal = hilbert(filt_signal)

        
    # amplitude_envelope = np.abs(analytic_signal) # not used
    phase = np.angle(analytic_signal)
    # instantaneous_frequency = (np.diff(instantaneous_phase) * (2.0*np.pi) * fs) # not used

    # filt = neo.AnalogSignal(signal = filt_signal, units ='Hz', sampling_rate = s.sampling_rate, 
    #                    annotations = s.annotations)#annotations['t_start'])

    return filt_signal, analytic_signal, phase


def read_LFP_from_exdir(group, original_exdir_path, cascade=True, lazy=False):
    """Copied and modified from exdirio.py"""
    from neo.core import AnalogSignal

    signal = group["data"]
    attrs = {}
    attrs.update(group.attrs.to_dict())
    attrs.update({'exdir_path': original_exdir_path})

    if lazy:
        ana = AnalogSignal([],
                           lazy_shape=(signal.attrs["num_samples"],),
                           units=signal.attrs["unit"],
                           sampling_rate=group.attrs['sample_rate'],
                           **attrs)
    else:
        ana = AnalogSignal(signal.data,
                           units=signal.attrs["unit"],
                           sampling_rate=group.attrs['sample_rate'],
                           **attrs)
    return ana

def load_LFP_from_exdir(ana, exdir_path, channel_group,
        high_res=False,allow_remove = False, **annotations):
    import copy 
    time_series_index = 1 if not high_res else 2

    import exdir.plugins.quantities

    exdir_file = exdir.File(exdir_path, mode ='w', plugins =
            [exdir.plugins.quantities], allow_remove = allow_remove)
    internal_path = 'processing/electrophysiology/channel_group_{}/LFP/LFP_timeseries_{}'.format(channel_group, time_series_index)
    group = exdir_file.require_group(internal_path)
    data = group.require_dataset('abc', data = 1)

    attrs = copy.deepcopy(ana.annotations)
    attrs.update(annotations)
    attrs.update({'name': ana.name,
                  'description': ana.description,
                  'start_time': ana.t_start,
                  'stop_time': ana.t_stop,
                  'sample_rate': ana.sampling_rate})
    group.attrs = attrs
    lfp_data = group.require_dataset('data', data=ana)
    lfp_data.attrs['num_samples'] = len(ana)
    lfp_data.attrs['sample_rate'] = ana.sampling_rate # TODO not save twice

def write_LFP_to_exdir(ana, file_path,  channel_group,
        high_res=False,allow_remove = False, **annotations):
    import copy 
    time_series_index = 1 if not high_res else 2

    import exdir.plugins.quantities

    exdir_file = exdir.File(file_path,  plugins =
            [exdir.plugins.quantities], allow_remove = allow_remove)
    internal_path = 'electrophysiology/channel_group_{}/LFP/LFP_timeseries_{}'.format(channel_group, time_series_index)
    processing_group = exdir_file.require_group('processing')
    group = processing_group.require_group(internal_path)

    attrs = copy.deepcopy(ana.annotations)
    attrs.update(annotations)
    attrs.update({'name': ana.name,
                  'description': ana.description,
                  'start_time': ana.t_start,
                  'stop_time': ana.t_stop,
                  'sample_rate': ana.sampling_rate})
    group.attrs = attrs
    lfp_data = group.require_dataset('data', data=ana)
    lfp_data.attrs['num_samples'] = len(ana)
    lfp_data.attrs['sample_rate'] = ana.sampling_rate # TODO not save twice
    return group
    


def get_sptr(exdir_path, unit_path, exdir_group=None, return_group = False,
        simple_structure=False):
    """Loads sptr data from exdir, either from an axona file structure or
    for a more simple customized structure. Can use already opened groups
    to reduce overhead"""
    if exdir_group is None:
        exdir_group = exdir.File(exdir_path)
    if simple_structure:
        unit_group = exdir_group[unit_path]
        sptr_data = unit_group['spike_times']
        metadata = {}
        times = pq.Quantity(sptr_data.data,
                            sptr_data.attrs['unit'])
        t_stop_attr = sptr_data.attrs['stop_time']
        t_start_attr = sptr_data.attrs['start_time']

        metadata.update(sptr_data.attrs.to_dict())
    else:
        sptr_group = exdir_group[unit_path]
        metadata = {}
        times = pq.Quantity(sptr_group['times'].data,
                            sptr_group['times'].attrs['unit'])
        t_stop_attr  = sptr_group.parent.attrs['stop_time']
        t_start_attr = sptr_group.parent.attrs['start_time']
        metadata.update(sptr_group['times'].attrs.to_dict())
    t_start = pq.Quantity(t_start_attr['value'], t_start_attr['unit'])
    t_stop = pq.Quantity(t_stop_attr['value'], t_stop_attr['unit'])

    metadata.update({'exdir_path': exdir_path})
    sptr = neo.SpikeTrain(times=times,
                      t_stop=t_stop,
                      t_start=t_start,
                      waveforms=None,
                      sampling_rate=None,
                      **metadata)
    if return_group:
        return sptr, exdir_group
    else:
        return sptr
    

def load_data(exdir_path, simple_structure = True, 
        data_path = '/home/halvard/cinpla/cinpla-work/exdir_data'):
    """Get data for a specific cell from exdir, either from an axona file structure or
    for a more simple customized structure. Can use already opened groups
    to reduce overhead. """
    split = exdir_path.split('/')
    rat_id =  split[1]
    unit_id = "unit{}".format(split[-1])
    channel_id = "channel{}".format(split[-3][-1])
    data_path = "{}/{}/main.exdir/".format(data_path,rat_id)
    unit_path = "{}/{}/".format(channel_id, unit_id)
    position_path = "position_data/"

    main_group = exdir.File(data_path)
    unit_group = main_group[channel_id][unit_id]
    sptr = get_sptr(data_path, unit_path, simple_structure=simple_structure)
    x,y,t,speed = get_pos_data(data_path, simple_structure=simple_structure)
    return x,y,t, speed ,sptr


def get_ith_rate_map(i, info,par , groups = {}, pos_data = {}, path_to_data_storage= "Data charlotte_pnn_mec/30rotter"):
    """Obsolete"""
    from itertools import islice
    idx, unit_info = islice(info.T.items(),i,None).__next__()
    
    exdir_path = path_to_data_storage + "/" +os.path.sep.join(unit_info.exdir_path.split('/')[:3])
    print("Getting" , unit_info.exdir_path)
    unit_path = '/'.join(unit_info.exdir_path.split('/')[3:])
    action_path = '/'.join(unit_info.exdir_path.split('/')[:3])

    groups.setdefault(action_path, )
    if action_path not in groups:
        groups[action_path] = exdir.File(exdir_path)
    exdir_group = groups.get(action_path)

    sptr = get_sptr(exdir_path, unit_path, exdir_group = exdir_group)

    if action_path not in pos_data:
        pos_data[action_path] = get_pos_data(exdir_path,par)
    x,y,t,speed = pos_data[action_path] 

    rate_map_raw = tr.spatial_rate_map(x, y, t, sptr, binsize=par['spat_binsize'],
                                   mask_unvisited=False, convolve = False)
    return rate_map_raw, unit_info.exdir_path

    
def next_rate_map(info, par, starting_index = 0, pos_data = {}, path_to_data_storage= "Data charlotte_pnn_mec/30rotter"):
    """Obsolete?"""
    for idx, unit_info in list(info.T.items())[starting_index:]:
        exdir_path = path_to_data_storage + "/" + os.path.sep.join(unit_info.exdir_path.split('/')[:3])
        action_path = '/'.join(unit_info.exdir_path.split('/')[:3])
        unit_path = '/'.join(unit_info.exdir_path.split('/')[3:])
        print("Getting" , unit_info.exdir_path)
        # x, y, t, ang, ang_t = tr.get_processed_tracking(exdir_path, par)
        sptr = get_sptr(exdir_path, unit_path)
        x,y,t,speed = pos_data.setdefault(action_path,
                get_pos_data(exdir_path,par))
        
        rate_map_raw = tr.spatial_rate_map(x, y, t, sptr, binsize=par['spat_binsize'],
                                       mask_unvisited=False,convolve = False)
        #smoothing=0.01, mask_unvisited=False, convolve = True)
        # rate_map = tr.spatial_rate_map(x, y, t, sptr, binsize=par['spat_binsize'],
                                       # smoothing=0.04, mask_unvisited=False, convolve = True) 
        yield idx, rate_map_raw, unit_info.exdir_path




def apply_smoothing(rate_map_raw, smoothing, box_xlen = 1*pq.m,
        binsize = 0.02*pq.m):
    from astropy.convolution import Gaussian2DKernel, convolve_fft
    rate_map_raw[np.isnan(rate_map_raw)] = 0.
    csize = (box_xlen / binsize) * smoothing
    kernel = Gaussian2DKernel(csize)
    rate_map = convolve_fft(rate_map_raw, kernel)  
    return rate_map


def get_field_statistics(rate_map, fields, nfields, par):
    from scipy.ndimage  import labeled_comprehension
    data = {}
    indx = np.arange(1,nfields+1)
    
    # per field:
    if nfields:
        data["field_sizes"]      = labeled_comprehension(rate_map, fields, indx, np.size, int, 0)
        data["field_mean_rates"] = labeled_comprehension(rate_map, fields, indx, np.mean,float, 0)
        data["field_max_rates"]  = labeled_comprehension(rate_map, fields, indx, np.max,float, 0)
    else:
        data["field_sizes"]      = np.nan
        data["field_mean_rates"] = np.nan
        data["field_max_rates"]  = np.nan
        
    # per unit:
    data["number_of_fields"] = nfields
    data["infield_rate"]     = np.mean(rate_map[fields > 0])
    data["outfield_rate"]    = np.mean(rate_map[fields == 0])
    data["avg_field_size"]   = np.mean(data['field_sizes'])
    #from exana.tracking import find_avg_size
    #data["TEST_field_size"]   = find_avg_size(rate_map,
            #binsize = par['spat_binsize'], box_xlen = par['box_xlen'],
            #box_ylen = par['box_ylen'])
    return data





def load_local_dataset(indata, par, raise_err = False):
    import exdir 
    from IPython.display import clear_output
    # exdir_path = "charlotte_pnn_mec/1627-081215-02/main.exdir/processing/electrophysiology/channel_group_1/UnitTimes/0"

    data = {}
    not_loaded = []
    errors = []
    N = len(indata)

    for i, (idx, item) in enumerate(indata.T.items()):
        clear_output(wait=True)
        print("{}/{}".format(i,N))
        exdir_path = item['exdir_path']
        print ("loading {}".format(exdir_path))
        
        try:
            x,y,t,speed,sptr = load_data(exdir_path)
            rm_raw =  tr.spatial_rate_map(x, y, t, sptr, binsize=par['spat_binsize'],
                                       mask_unvisited=False, convolve = False)
            data[exdir_path] = [x,y,t,speed,sptr,rm_raw]
        except (FileNotFoundError, KeyError) as err:
            not_loaded.append(exdir_path)
            if raise_err:
                raise
            
    print("{}/{} data loaded succesfully".format(N-len(not_loaded), N))
    return data, not_loaded, errors

def isi_statistics(sptr):
    import elephant.statistics as ep_stats
    isi = ep_stats.isi(sptr)
    std = np.std(isi)
    mean = np.mean(isi)
    cv = np.float64(std/mean)
    return cv,std,mean,isi



def spike_selector(sptr, times ):
    dt = np.mean(np.diff(times))
    spikes_in_times = sptr[np.logical_and(sptr > times[0], sptr < times[-1]+dt)]
    return spikes_in_times

def get_channel_number(exdir_path):
    phrase = 'channel_group_'
    i = exdir_path.find('channel_group')
    return int(exdir_path[i+len(phrase)])


    
def get_LFP_spectrum(anas, chunksize = 1*pq.s, nchunks = 1000, sample_rate =
        250*pq.Hz, z_normalized = True, 
        old_crappy_method = False # for helvete
        ):
    from scipy import signal

    fs = sample_rate
    samples_per_chunk = int(fs.rescale('Hz').magnitude*chunksize.rescale('s').magnitude)
    freq_range = [7,12]

    
    if len(anas) > 1:
        anas_arrays = np.concatenate([a.magnitude[:,0] for a in anas if
            a.annotations['sample_rate'] == sample_rate]).reshape(2,-1)
    else:
        anas_arrays = anas[0].magnitude[:,0] 
    
    if z_normalized:
        anas_arrays = ((anas_arrays.T -
                anas_arrays.mean(axis=1))/anas_arrays.std(axis=1)).T
    if old_crappy_method:
        random_intervals = np.random.randint(anas_arrays.shape[1]-samples_per_chunk, size=nchunks)

        samples = []
        for r in random_intervals:
            samples = np.concatenate((samples, np.arange(r, r+samples_per_chunk, dtype=int))).astype(int)
            
        anas_chunks = anas_arrays[:, samples]
        fpre, Pxxpre = signal.welch(anas_chunks, fs, nperseg=1024)
    else:
        fpre, Pxxpre = signal.welch(anas_arrays, fs, nperseg=1024)
    return fpre, Pxxpre

