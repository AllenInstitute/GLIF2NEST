import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import random


def get_step_trace(step_times, step_currs, dt_ms, sim_time_ms, default_curr=0.0):

    assert(len(step_times) == len(step_currs))

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

def get_step_trace_noise(step_times, step_currs, dt_ms, sim_time_ms, default_curr=0.0):

    assert(len(step_times) == len(step_currs))

    noise_level=0.2
    n_times = int(sim_time_ms/float(dt_ms))

    I = [default_curr for _ in xrange(n_times)]

    #convert step_times[i] to correct I, and set I[step_times[i]:step_times[i+1]] = step_currs[i]
    for i in xrange(len(step_times) - 1):
        period_begin = int(step_times[i]/float(dt_ms)) # get the current window frame array location
        period_end = int(step_times[i+1]/float(dt_ms))
        period_len = period_end - period_begin
        if step_currs[i]==0.0:
            I[period_begin:period_end] = [step_currs[i]]*period_len
        else:
            I[period_begin:period_end] = [step_currs[i]+random.uniform(-step_currs[i]*noise_level,step_currs[i]*noise_level) for _ in xrange(period_len)]

    # last current windows
    period_begin = int(step_times[-1]/float(dt_ms))
    period_len = len(I) - period_begin
    if step_currs[-1]==0.0:
        I[period_begin:] = [step_currs[-1]]*period_len
    else:
        I[period_begin:] = [step_currs[i]+random.uniform(-step_currs[i]*noise_level,step_currs[i]*noise_level) for _ in xrange(period_len)]
    
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


def plt_comparison_neurons(I, neurons_ms, neurons_v, neurons_spikes_ms, show=False):
    def ylim(vm, margins=.1):
        bt = (1 - (1 - margins))/float(1 - margins)
        margin = bt*(max(vm) - min(vm))
        return (min(vm) - margin, max(vm) + margin)

    num_neurons=len(neurons_spikes_ms)
    gs = gridspec.GridSpec(num_neurons+2, 1, height_ratios=[2]+[4 for i in range(num_neurons)]+[2])

    # Plot the spike trains
    ax1 = plt.subplot(gs[0])
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    pltcltypes=['b','r','g','m','c','k','y']
    #pltcltypes=colors.cnames.values() # full list of colors
    for i in range(num_neurons):
        plt.plot(neurons_spikes_ms[i], [0.2*(i+1)] * len(neurons_spikes_ms[i]), '.', color=pltcltypes[i])        
    plt.xlim(neurons_ms[0][0], neurons_ms[0][-1])
    plt.ylim(0, 1)

    # Plot the voltage traces of neurons
    for i in range(num_neurons):
        ax2 = plt.subplot(gs[num_neurons-i])
        ax2.axes.get_xaxis().set_visible(False)
        ax2.set_ylabel('V (mV)', fontsize=10)
        plt.xlim(neurons_ms[i][0], neurons_ms[i][-1])
        plt.ylim(*ylim(neurons_v[i], margins=.05))
        l2, = plt.plot(neurons_ms[i], neurons_v[i], pltcltypes[i], label='neuron'+str(i)+': '+str(len(neurons_spikes_ms[i])))
        leg = plt.legend(handles=[l2], loc=2)
        plt.gca().add_artist(leg)

    # Plot the currents
    ax3 = plt.subplot(gs[num_neurons+1])
    ax3.set_xlabel('t (ms)', fontsize=10)
    ax3.set_ylabel('I (pA)', fontsize=10)
    plt.plot(neurons_ms[0], I[:-1])
    plt.xlim(neurons_ms[0][0], neurons_ms[0][-1])
    plt.ylim(*ylim(I, margins=.2))

    if show:
        plt.show()
        
def plt_comparison_2neurons(I, neuron1_ms, neuron1_v, neuron1_spikes_ms, neuron2_ms, neuron2_v, neuron2_spikes_ms, show=False):
    def ylim(vm, margins=.1):
        bt = (1 - (1 - margins))/float(1 - margins)
        margin = bt*(max(vm) - min(vm))
        return (min(vm) - margin, max(vm) + margin)

    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 4, 4, 1])

    # Plot the spike trains
    ax1 = plt.subplot(gs[0])
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    plt.plot(neuron1_spikes_ms, [0.2] * len(neuron1_spikes_ms), '.b')
    plt.plot(neuron2_spikes_ms, [0.8] * len(neuron2_spikes_ms), '.r')
    plt.xlim(neuron1_ms[0], neuron1_ms[-1])
    plt.ylim(0, 1)

    # Plot the voltage traces of neuron2
    ax2 = plt.subplot(gs[1])
    ax2.axes.get_xaxis().set_visible(False)
    ax2.set_ylabel('V', fontsize=10)
    plt.xlim(neuron2_ms[0], neuron2_ms[-1])
    plt.ylim(*ylim(neuron2_v, margins=.05))
    l2, = plt.plot(neuron2_ms, neuron2_v, '--r', label='neuron2')
    leg = plt.legend(handles=[l2], loc=2)
    plt.gca().add_artist(leg)
    
    # Plot the voltage traces of neuron1
    ax2 = plt.subplot(gs[2])
    ax2.axes.get_xaxis().set_visible(False)
    ax2.set_ylabel('V', fontsize=10)
    plt.xlim(neuron1_ms[0], neuron1_ms[-1])
    plt.ylim(*ylim(neuron1_v, margins=.05))
    l1, = plt.plot(neuron1_ms, neuron1_v, 'b', label='neuron1')
    leg = plt.legend(handles=[l1], loc=2)
    plt.gca().add_artist(leg)


    # Plot the currents
    ax3 = plt.subplot(gs[3])
    ax3.set_xlabel('t (ms)', fontsize=10)
    ax3.set_ylabel('I', fontsize=10)
    plt.plot(neuron1_ms, I[:-1])
    plt.xlim(neuron1_ms[0], neuron1_ms[-1])
    plt.ylim(*ylim(I, margins=.2))

    if show:
        plt.show()
        
def plt_comparison(I, allen_ms, allen_v, allen_spikes_ms, nest_ms, nest_v, nest_spikes_ms, show=False):
    def ylim(vm, margins=.1):
        bt = (1 - (1 - margins))/float(1 - margins)
        margin = bt*(max(vm) - min(vm))
        return (min(vm) - margin, max(vm) + margin)

    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 6, 1])

    # Plot the spike trains
    ax1 = plt.subplot(gs[0])
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    plt.plot(allen_spikes_ms, [0.8] * len(allen_spikes_ms), '.b')
    plt.plot(nest_spikes_ms, [0.2] * len(nest_spikes_ms), '.r')
    plt.xlim(allen_ms[0], allen_ms[-1])
    plt.ylim(0, 1)

    # Plot the voltage traces
    ax2 = plt.subplot(gs[1])
    ax2.axes.get_xaxis().set_visible(False)
    ax2.set_ylabel('V', fontsize=10)
    plt.xlim(allen_ms[0], allen_ms[-1])
    plt.ylim(*ylim(allen_v, margins=.05))
    l1, = plt.plot(allen_ms, allen_v, 'b', label='allen')
    l2, = plt.plot(nest_ms, nest_v, '--r', label='nest')
    leg = plt.legend(handles=[l1, l2], loc=2)
    plt.gca().add_artist(leg)

    # Plot the currents
    ax3 = plt.subplot(gs[2])
    ax3.set_xlabel('t (ms)', fontsize=10)
    ax3.set_ylabel('I', fontsize=10)
    plt.plot(allen_ms, I)
    plt.xlim(allen_ms[0], allen_ms[-1])
    plt.ylim(*ylim(I, margins=.2))

    if show:
        plt.show()