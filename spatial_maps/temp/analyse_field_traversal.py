import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import interpolate
# import quantities as pq
# import exana.tracking as tr
import pandas as pd


def analyse_phase_precession_old(sp_x, sp_y, x_c,y_c, sptr, path_dist, spike_phases,
        size): # = np.sqrt(2)*pq.m
    """Plots spikes in bins. Bins are defined by some distance measurement
    path_dist.
    
    Parameters:
    -----------
    
    Returns:
    --------
    
    
    """
    # TODO: Test if units are correct/if there are units at all 

    from scipy import interpolate
    

    dist = np.sqrt((sp_x - x_c)**2+(sp_y - y_c)**2)
    #angle =  np.arctan2(y_pos - center[1], sp_x - center[0])
    mask = dist < size
    if np.sum(mask) == 0:
        raise RuntimeError('found no spikes within dist size of field center')
    path_dist = path_dist[mask]
    spike_phases = spike_phases[mask]

    pathdist_bin = np.linspace(-size.magnitude,size.magnitude,200000)*size.units
    spikes_per_pathdist_bin, _ = np.histogram(path_dist, pathdist_bin)
    pdi = np.digitize(path_dist,pathdist_bin)
    phase_per_bin = np.zeros(pathdist_bin.shape) * spike_phases.units


    for n in range(pdi.size ):
        phase_per_bin[pdi[n]-1] += spike_phases[n] %(2*np.pi*spike_phases.units) 

    phase_per_bin = np.divide(phase_per_bin[:-1],spikes_per_pathdist_bin)

    mask = [~np.isnan(phase_per_bin)] 
    bins = pathdist_bin[:-1][mask]
    phases = phase_per_bin[mask]

    phases *= 180/np.pi

    a,b = np.polyfit(bins,phases, 1)

    c = (sptr.magnitude/sptr.t_stop.magnitude)
    import matplotlib.pyplot as plt
    plt.scatter(bins,phases)
    #plt.scatter(pathdist_bin[:-1], phase_per_bin)#, c=c)
    plt.plot(bins , bins*a+b*bins.units)
    return plt.gca()




def project_bc_dist_to_vel():
    """ 
    returns the projection of distance to center coordinates onto the unit
    velocity vector  
    """
    speed = np.sqrt(vx**2 + vy**2)

    # unit velocity
    u_v = np.array((vx/speed,vy/speed)) 
    pos = np.array((x-x_c, y-y_c)) 
    d = np.sum(u_v*pos, axis=0)
    return d


def distance_to_edge_function(x_c, y_c, field, box_xlen, box_ylen,interpolation = 'linear'):
    """Returns a function which for a given angle returns the distance to
    the edge of the field from the center.
    Parameters:
        x_c
        y_c
        field: numpy 2d array
            ones at field bins, zero elsewhere
    """

    valid_methods = ['lapl','plt_cont','_cntr', 'scikit']

    from skimage import measure
    contours = measure.find_contours(field, 0.5) 
        raise(ValueError('invalid argument, '))
    
    print(contours)
    box_dim = np.array([box_xlen, box_ylen])
    edge_x, edge_y = (contours[0]*box_dim/(np.array(field.shape)-(1,1))).T 

    # angle between 0 and 2\pi
    angles = np.arctan2((edge_y - y_c), (edge_x - x_c))%(2*np.pi) 
    a_sort = np.argsort(angles)
    angles = angles[a_sort]

    distances = np.sqrt((edge_x-x_c)**2 + (edge_y-y_c)**2)
    distances = distances[a_sort]

    N = len(edge_x)

    # Fill in edge values for the interpolation
    pad_a = np.pad( angles, 2, mode = 'linear_ramp',
            end_values = (0,2*np.pi))
    ev = (distances[0] + distances[1])/2
    pad_d = np.pad(distances, 2, mode = 'linear_ramp',
            end_values=ev)

    if interpolation=='cubic':
        mask = np.where(np.diff(pad_a) == 0)
        pad_a = np.delete(pad_a, mask)
        pad_d = np.delete(pad_d, mask)
    dist_func = interpolate.interp1d(pad_a,pad_d,kind=interpolation)
    
    return dist_func


def map_pass_to_unit_circle(x,y,t, x_c, y_c, field=None, dist_func=None):
    """Uses three vectors {v,p,q} to map the passes to the unit circle. v
    is the average velocity vector of the pass, p is the vector from the
    position (x,y) to the center of the field and q is the vector from the
    center to the edge through (x,y). Inspired by [1].
    
    Parameters:
    -----------
        :x, y, t: np arrays
            should contain x,y and t data in numpy arrays
        :x_c , y_c: floats
            bump center
        :field: np 2d array
            bins indicating location of field.
        :dist_func: function
            dist_func(angle) = distance to bump edge from center
        :return_vecs(optional): bool, default False

    Returns:
    --------
        r : arrays of distance to origin on unit circle
        theta : arrays of angles to axis defined by mean velocity vector

    References:
    -----------
        [1]: A. Jeewajee et. al., Theta phase precession of grid and placecell firing in open
        environment
    """
    pos = np.array((x,y))

    # vector from pos to center p
    p_vec = ((x_c,y_c) - pos.T).T 
    # angle between x-axis and negative vector p
    angle = (np.arctan2(p_vec[1],p_vec[0]) + np.pi)%(2*np.pi)
    # distance from center to edge at each angle
    q = dist_func(angle)
    # distance from center to pos
    p = np.linalg.norm(p_vec,axis=0)
    # r-coordinate on unit circle
    r = p/q

    dpos = np.gradient(pos, axis = 1)
    dt   = np.gradient(t.magnitude)
    vel  = np.divide(dpos,dt)

    # mean velocity vector v
    v_vec = np.average(vel,axis = 1)
    # angle on unit circle, run is rotated such that mean velocity vector
    # is toward positive x
    theta = (angle - np.arctan2(v_vec[1],v_vec[0]))%(2*np.pi)
    return r, theta

    
        
def get_pass_data(x, y, t, sptr, fields, bump_centers, box_xlen, box_ylen,
        phase_func=lambda x:0, speed_thrsh = 0, 
        dist_func_method = '_cntr'):
    """Gets data for each of the passes 
    Parameters
    ----------
    x,y,t : 
        position data
    sptr : 
        spike times
    fields : 
        labeled fields
    bump_centers: (Nx2) quantities array
        centers of bumps
    speed_thrsh : float, default 0
        lower limit on minimum of speed of a pass to be marked valid
    phase_func : function
        function that for a given time gives the phase of the local field
        potential. Default lambda x:0     
   

    Returns:
    -------
        passes : pandas DataFrame
            contains information about each pass.
            columns=[field,x,y,t,r,theta,pdcd,spikes,phases,spike_x,spike_y]
            (at least, may have forgotten some)
        """
    import scipy.interpolate


    passes = field_traversals(x,y,t,fields, box_xlen = box_xlen, box_ylen = box_ylen)
    dist_funcs = []

    # TODO : Get the index from the existing fields
    index = np.arange(1,np.max(fields)+1)

    for ind in index:
        field = fields == ind
        x_c,y_c = bump_centers[ind-1]
        df = distance_to_edge_function(x_c,y_c,field, 
                box_xlen = box_xlen, box_ylen = box_ylen,
                method = dist_func_method, res_exp=1)
        dist_funcs.append(df)

    for pass_info in passes:
        ind = pass_info['field']
        x_c,y_c = bump_centers[ind-1]
        pass_times = pass_info['t']
        pass_x = pass_info['x']
        pass_y = pass_info['y']

        if not speed_thrsh:
            is_valid = True
        elif pass_times.size < 2:
            is_valid = False
        else:
            vx = np.gradient(pass_x) / np.gradient(pass_times)
            vy = np.gradient(pass_y) / np.gradient(pass_times)

            vel = np.linalg.norm((vx,vy), axis=0)
            # np.min removes units...
            thrsh = speed_thrsh.rescale(vx.units).magnitude
            is_valid = np.min(vel) > thrsh

        df = dist_funcs[ind-1]

        # spike info
        pass_spikes = spikes_in_passes([pass_times], sptr)[0]
        pass_phases = phase_func(pass_spikes) 

        # unit circle parameters for the run:
        pass_r, pass_theta, out_data = map_pass_to_unit_circle(
                pass_info['x'], pass_info['y'], pass_info['t'], 
                x_c, y_c, df, return_pdcd = True, return_pdmd=True, return_unit_vel = True,
                return_vecs=True)


        pass_info['r'] = pass_r
        pass_info['theta'] = pass_theta
        pass_info.update(out_data)# ['pdcd'] = pass_pdcd
        pass_info['spikes'] = pass_spikes
        pass_info['phases'] = pass_phases
        pass_info['dist_func'] = df
        pass_info['valid'] = is_valid
        pass_info['bc'] = bump_centers[ind-1]
    return pd.DataFrame(passes)



def simulate_pass_sptr_with_theta(passes, times, filt_sig, n_samples=10):
    """Uses an nonhomogeneous poisson process modulated by filt_sig to
    generate n_samples spike trains for each pass defined in passes. 
    NB: passes is a dataframe, return is a data_frame. times is the times
    of filt_sig, used to modulate
    inhomogeneous poisson process"""
    N  = len(passes)

    from IPython.display import clear_output, display
    import neo
    import elephant.spike_train_generation as stg

    sampling_rate = 1/np.mean(np.diff(times))
    print(sampling_rate)
    sim_sptr_data = {}

    for i,(ind,p) in enumerate(passes.T.items()):
        clear_output(wait=True)
        display(f'{i+1}/{N}')
        t = p['t']

        pass_sptr = p['spikes']
        mask = np.logical_and(times > t[0], times <= t[-1])
        sim_rate = filt_sig.copy()[mask]

        thrsh = 0
        sim_rate -= np.min(sim_rate)
        sim_rate /= np.max(sim_rate)

        rate = np.size(pass_sptr) / (t[-1] - t[0])

        sim_rate *= 2*rate 
        rate_array = neo.AnalogSignal(sim_rate, units = 'Hz', sampling_rate
                = sampling_rate, t_start=t[0])
        # p['sim_sptr'] = []
        sim_sptr_data[ind] = []
        for i in range(n_samples):
            if np.size(pass_sptr) == 0: # cheap fix way of adding empty sptrs
                # p['sim_sptr'].append(pass_sptr.copy())
                sim_sptr_data[ind].append(pass_sptr.copy())
            else:
                sim_sptr = stg.inhomogeneous_poisson_process(rate_array)
                sim_sptr_data[ind].append(sim_sptr)

    passes['sim_sptr'] = pd.Series(sim_sptr_data)
    return passes


def interspike(x,lamb):
    """an interspike function, parameter lamb is the fit parameter"""
    f = x*np.exp(-x*lamb)
    return f/np.sum(f)

def fit_func_to_hist(data, binwidth = None, nbins = 10, func = interspike):
    """fits an function to a histogram of data. 
    
    Returns
    =======
    parameters, 
    covariance matrix, 
    bin edges,
    normalized bin counts."""
    from scipy.optimize import curve_fit

    if binwidth is None:
        entries, bin_edges = np.histogram(data, bins = np.linspace(int(np.min(diff_ms)), np.max(diff_ms),nbins))
    else:
        entries, bin_edges = np.histogram(data, bins = np.arange(int(np.min(data)), np.max(data),binwidth))

    s = np.sum(entries)
    entries = entries / s
    # calculate binmiddles
    bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])


    # fit with curve_fit
    parameters, cov_matrix = curve_fit(func, bin_middles, entries) 

    return parameters, cov_matrix, bin_middles, bin_edges, entries



def plot_passes(passes, field_index=None, plot_all=False,
        plot_invalid=False, plot_original = False):
    fig_main, (ax_left,ax_mid, ax_right) = plt.subplots(1,3)

    from scipy.interpolate import interp1d
    cmap = plt.cm.viridis
    import cmocean 
    circ_cmap  = cmocean.cm.phase
    a_ = np.linspace(0,2*np.pi,200)
    ax_left.plot(np.cos(a_),np.sin(a_))

    maxspikes = np.max([len(p['spikes']) for p in passes])
    max_index = np.max([p['field'] for p in passes])

    for pass_info in passes:
        field = pass_info['field']

        # continue if field unequal to a specified field_index
        if (not (field_index is None)) and (field != field_index):
            continue

        spikes = pass_info['spikes']

        if (spikes.size == 0 or not pass_info['valid']) and not plot_invalid:
            continue
        x_pos = pass_info['x']
        y_pos = pass_info['y']
        t = pass_info['t']
        bc = pass_info['bc']
        spikes = pass_info['spikes']
        phases = pass_info['phases']
        r = pass_info['r']
        theta = pass_info['theta']
        pdcd = pass_info['pdcd']
        df = pass_info['dist_func']

        if t.size <= 2:
            continue

        sp_pdcd  = interp1d(t,pdcd,bounds_error = False,
                fill_value=(np.min(pdcd),np.max(pdcd)))(spikes)
        sp_theta = interp1d(t,theta)(spikes)
        sp_r     = interp1d(t,r)(spikes)

        # plot passes and spikes on unit circle
        c = cmap(field/max_index)
        #scat_c = cmap((sp_pdcd+1)/2)
        ax_left.plot(r * np.cos(theta),r * np.sin(theta),c=c,zorder=0)
        scat_c = circ_cmap(phases%(2*np.pi)/(2*np.pi))

        ax_left.scatter(sp_r*np.cos(sp_theta),
                sp_r*np.sin(sp_theta),c=scat_c, zorder=1)
        # ax1.scatter(pr*)
        # ax1.plot(p_pos_r*np.cos(p_pos_theta), p_pos_r*np.sin(p_pos_theta))
        # ax1.scatter(pr*np.cos(pthet),pr*np.sin(pthet), c=c)
        # ax11.scatter(pr*np.cos(pthet),pr*np.sin(pthet), c=c)


        x_var1 = ((spikes.times- t[0])/(t[-1]-t[0])).magnitude
        y_var1 = phases*180/np.pi

        x_var2 = sp_r*np.cos(sp_theta)
        y_var2 = phases*180/np.pi


        xlim = 0.15
        ylim = np.pi
        spike_lim = 5

        c2 = plt.cm.jet(len(spikes)/maxspikes)
        # c2 = c
        if spikes.size > spike_lim:
            a2,b2 = np.polyfit(x_var2,y_var2,1)
            a1,b1 = np.polyfit(x_var1,y_var1,1)
            ax_right.plot(x_var2, a2*x_var2 + b2, c=c2)
            ax_mid.plot(x_var1, a1*x_var1 + b1, c = c2)

        ax_right.scatter(x_var2,y_var2, c=scat_c)
        ax_mid.scatter(x_var1,y_var1, c=c)

        if plot_all:
            if plot_original:
                fig = plt.figure()
                ax1_2 = fig.add_subplot(231)
                ax1 = fig.add_subplot(234)
                ax2 = fig.add_subplot(132)
                ax3 = fig.add_subplot(133)

                a_ = np.linspace(0,2*np.pi,200)
                dist_ = df(a_)

                ax1_2.plot(bc[0].magnitude+dist_*np.cos(a_) ,
                           bc[1].magnitude+dist_*np.sin(a_))
                ax1_2.plot(x_pos, y_pos)
                ax1_2.scatter(bc[0],bc[1])

                # feels like a waste to do this for each pass...:
                x_interp = interp1d(t,x_pos)
                y_interp = interp1d(t,y_pos)
                sp_x = x_interp(spikes)
                sp_y = y_interp(spikes)

                ax1_2.scatter(sp_x, sp_y)
            else:
                fig, axes = plt.subplots(1,3)    
                ax1,ax2,ax3 = axes
            ax1.plot(np.cos(a_),np.sin(a_))
            ax1.plot(r*np.cos(theta),r*np.sin(theta)) 
            ax1.scatter(sp_r*np.cos(sp_theta),
                sp_r*np.sin(sp_theta),c=scat_c)
            ax1.axis('equal')

            if spikes.size > spike_lim:
                ax2.plot(x_var1, a1*x_var1 + b1)
                ax3.plot(x_var2, a2*x_var2 + b2)

            ax2.scatter(x_var1,y_var1, c=c)
            ax2.axis([0,1,-180,180])
            ax3.scatter(x_var2,y_var2, c=c2)
            #ax3.axis([1,-1,-180,180])
            #ax3.reverse_axis()
            fig.suptitle('field %d' %field)

    ax_left.axis('equal')
    ax_mid.axis([0,1,-180,180])
    #ax_right.axis([1,-1,-180,180])




