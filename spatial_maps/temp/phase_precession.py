import numpy as np
import quantities as pq
from scipy.interpolate import interp1d
from . import analyse_field_traversal as aft


def create_passes(x,y,t,fields,bc, spike_rate = 9*pq.Hz, LFP_rate = 8*pq.Hz,
                 spike_time_noise = 0):
    """Creates passes with regular spiking according to spike_rate,
    starting at the first peak of the LFP phase"""
    hr_t = np.linspace(t[0],t[-1],100000)
    LFP = np.sin(LFP_rate.magnitude*2*np.pi*hr_t) + 1j*np.cos(LFP_rate.magnitude*2*np.pi*hr_t)
    phase = np.angle(LFP)

    xfunc  = interp1d(t,x)
    yfunc  = interp1d(t,y)
    phasefunc = interp1d(hr_t, phase)
    funcs = [xfunc, yfunc, phasefunc]


    from scipy import ndimage
    in_field = aft.in_field(x*pq.m,y*pq.m,fields)
    lbl, nlbl = ndimage.label(in_field)
    indx = np.arange(nlbl)+1
    sizes = ndimage.labeled_comprehension(lbl, lbl, indx, np.size, int, 0)

    # only keep passes longer than 0.1 sec ( no noise-passes )
    # (assumes 100Hz rec frequency)
    valid_passes = sizes > 10
    
    num = 1
    for i,ind in enumerate(indx):
        if not valid_passes[i]:
            lbl[lbl == ind] = 0
        else:
            lbl[lbl == ind] = num
            num += 1
            
    slices = ndimage.find_objects(lbl)

    spike_times = []
    for o in slices: # in valid_passes:
        t_enter, t_exit = t[o][[0,-1]]

        # Start spike generation when LFP phase is 0
        mask = hr_t > t_enter
        mod_phase = (phase%(2*np.pi))[mask]
        #import matplotlib.pyplot as plt
        #plt.figure()
        #plt.plot(np.diff(mod_phase))
        t_spike_start = hr_t[mask][np.where(np.diff(mod_phase) > np.pi)[0][0]]
        # print(t_spike_start - t_enter)
        spike_times.append(np.arange(t_spike_start, t_exit, 1/spike_rate.magnitude))
        
    spike_times = np.concatenate(spike_times)
    spike_times += np.random.normal(0,spike_time_noise, size=len(spike_times))

    sp_x = xfunc(spike_times)
    sp_y = yfunc(spike_times) 
    sp_phase = phasefunc(spike_times)

    passes = aft.get_pass_data(x*pq.m,y*pq.m,t*pq.s,spike_times*pq.s, fields, bc,phasefunc, dist_func_method = 'scikit')
    return passes, funcs


def plot_passes(passes, fields, funcs, x=None, y=None, track_alpha = None,
        extent = [0,1,0,1]):
    # box = [[0,1,1,0,0],[0,0,1,1,0]]
    box = [[extent[0],extent[1],extent[1],extent[0],extent[0]],
            [extent[2],extent[2],extent[3],extent[3],extent[2]]]
    import matplotlib.pyplot as plt
    xfunc, yfunc, phasefunc = funcs
    
    fig = plt.figure(figsize=[16,12])

    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(323)
    ax3 = fig.add_subplot(322)
    ax4 = fig.add_subplot(324)
    ax5 = fig.add_subplot(313)
    
    m = ax1.scatter([0,0],[0,0],c = [-np.pi, np.pi])
    cax = plt.colorbar(m, ax=ax1)
    cax.set_label('phase')
    delta_f_values = []

    for i, p in passes.T.items():
    # plot pass
        sp_x = xfunc(p.spikes)
        sp_y = yfunc(p.spikes)
        sp_pdcd = interp1d(p.t, p.pdcd)(p.spikes)
        sp_pdmd = interp1d(p.t, p.pdmd)(p.spikes)
        sp_t = p.spikes
        sp_phase = p.phases

        ax1.plot(p.x,p.y)
        ax1.contour(fields.T>0,1, colors='k', extent = extent, origin='lower')
        if not (x is None or y is None):
            alpha = track_alpha or len(p.x)/len(x) 
            ax1.plot(x,y, 'k', alpha=len(p.x)/len(x))
        m = ax1.scatter(sp_x,sp_y, c = sp_phase)
        ax1.plot(*box)
        ax1.set_title('Box, field, pass and spikes')
        ax1.set_xlabel('x [m]')
        ax1.set_ylabel('y [m]')
        ax1.axis('equal')
        # ax1.set_xlabel('y')

        sp_r     = interp1d(p.t, p.r)(sp_t)
        sp_theta = interp1d(p.t, p.theta)(sp_t)
        
        # plt.contour(fields,1, colors='k', extent = [0,1,0,1], origin='lower')
        ax2.plot(p.r*np.cos(p.theta), p.r *np.sin(p.theta), label = i)
        ax2.scatter(sp_r*np.cos(sp_theta), sp_r *np.sin(sp_theta))
        # plt.scatter(sp_x,sp_y, c = sp_phase)
        temp = np.linspace(0,2*np.pi,200)
        unit_circ = [np.cos(temp), np.sin(temp)]
        ax2.plot(*unit_circ, color = 'k')
        ax2.set_title('Pass mapped to unit circle')
        ax2.axis('equal')

        # plot phases
        pdcd = False
        if pdcd :
            ax3.scatter(sp_pdcd, p.phases)
            ax3.set_xlabel('dist to peak projected onto the current direction [pdcd]', size =20)
        else:
            ax3.scatter(sp_pdmd, p.phases)
            ax3.set_xlabel('dist to peak projected onto the mean direction [pdmd]', size =20)

        ax3.set_ylabel('phase [rad]', size =20)
        # print('Spike phases: ',p.phases)
        # print('Spike times: ', sp_t)

        delta_f = (np.diff(p.phases)%(2*np.pi)/(2*np.pi))/np.diff(sp_t)
        delta_f_values.append(delta_f)
        #print('delta f:', delta_f)

        #print(sp_t[0]) 
        #print(p.t[0])
        ax4.scatter(sp_t.magnitude - p.t[0].magnitude, p.phases)
        # ax4.plot(t, phasefunc(t))
        ax4.set_xlabel('spike time [ s]', size =20)
        ax4.set_ylabel('phase [rad]', size =20)
        
        ax5.plot(delta_f, 'o')
        ax5.axhline(np.mean(delta_f))
        ax5.set_xlabel('isi num')
        ax5.set_ylabel('calculated frequency difference')
        # ax5.set_ylim([0,1.1*np.max(delta_f).magnitude])
        # fig.tight_layout()
    ax2.legend()
    return delta_f_values
    
