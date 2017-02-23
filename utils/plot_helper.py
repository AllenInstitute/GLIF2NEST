import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def get_step_trace(step_times, step_currs, dt_ms, sim_time_ms, default_curr=0.0):
#def step_generator_trace(step_times, step_currs, sim_times, default_curr=0.0):
    assert(len(step_times) == len(step_currs))
    #assert(len(sim_times) > 1)
    
    #n_times = len(sim_times)
    #dt = sim_times[1] - sim_times[0] # Assume dt is constant over simulation

    n_times = int(sim_time_ms/float(dt_ms))

    I = [default_curr for _ in xrange(n_times)]

    #convert step_times[i] to correct I, and set I[step_times[i]:step_times[i+1]] = step_currs[i]
    for i in xrange(len(step_times) - 1):
        period_begin = int(step_times[i]/float(dt_ms)) # get the current window frame array location
        period_end = int(step_times[i+1]/float(dt_ms))
        period_len = period_end - period_begin
        I[period_begin:period_end] = [step_currs[i]]*period_len

    # last current windows
    period_begin = int(step_times[-1]/float(dt_ms))
    period_len = len(I) - period_begin
    I[period_begin:] = [step_currs[-1]]*period_len
    return I

def step_generator_trace(step_times, step_currs, sim_times, default_curr=0.0):
    assert(len(sim_times) > 1)

    dt = sim_times[1] - sim_times[0]
    return get_step_trace(step_times, step_currs, dt, sim_times[-1], default_curr)



def plot_vt(ts, voltages, I, show=True):
    def ylim(vm):
        # Add margins to y
        b = .10 # percentage of bottom + top margins
        bt = (1 - (1 - b))/float(1 - b)
        margin = bt*(max(vm) - min(vm))
        return (min(vm) - margin, max(vm) + margin)

    gs = gridspec.GridSpec(2, 1, height_ratios=[7,1])
    ax1 = plt.subplot(gs[0])
    ax1.axes.get_xaxis().set_visible(False)
    ax1.set_ylabel('V', fontsize=20)
    plt.plot(ts, voltages)
    plt.ylim(*ylim(voltages))

    ax2 = plt.subplot(gs[1])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    if min(I) != max(I):
        ax2.set_yticks(np.arange(min(I) - 1, max(I) + 1, max(I) - min(I) + 1))
    else:
        ax2.set_yticks([min(I)])
    ax2.set_xlabel('t', fontsize=20)
    ax2.set_ylabel('I', fontsize=20)
    plt.plot(ts, I)
    if show:
        plt.show()
