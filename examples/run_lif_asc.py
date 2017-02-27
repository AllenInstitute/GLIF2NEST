import os
import sys
import nest
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from allensdk.model.glif.glif_neuron import GlifNeuron
import allensdk.core.json_utilities as json_utilities
import matplotlib.gridspec as gridspec

sys.path.append(os.path.join(sys.path[0], '../utils'))
from plot_helper import plot_vt, step_generator_trace, get_step_trace, plt_comparison
import allensdk_helper as asdk


nest.Install('glifmodule.so')

#neuron = nest.Create('glif_lif_asc')
#print nest.GetStatus(neuron, 'asc_init')
#nest.SetStatus(neuron, {'asc_init': [1.0, 2.0]})
#print nest.GetStatus(neuron, 'asc_init')

def run_lif_asc_allensdk(cell_id, model_type, I, dt_ms, base_dir='../models'):
    # Get the neuron configuration file
    m_data = asdk.get_models_dir(base_dir=base_dir, cell_id=cell_id, model_type=model_type)
    if len(m_data) == 0:
        print("Couldn't find a {} model for cell {} in {}.".format(model_type, cell_id, base_dir))
        return

    if len(m_data) > 1:
        print('Found more than one {} model for cell {}.'.format(model_type, cell_id))
        return

    # Read the configuration file
    model_files = m_data[0]
    model_config_file = model_files['model-config-file']
    neuron_config = json_utilities.read(model_config_file)
    neuron = GlifNeuron.from_dict(neuron_config)
    neuron.dt = dt_ms * 1.0e-04 # convert from miliseconds to seconds

    # Run the simulation
    output = neuron.run(I)

    # Return run-time, voltage trace and spike times
    voltages = output['voltage']
    times = [t*dt_ms for t in xrange(len(I))]
    spike_times = output['interpolated_spike_times'] * 1.0e04 # return spike-times in miliseconds
    return times, voltages, spike_times


def run_lif_asc_nest(cell_id, amp_times, amp_vals, dt_ms, simulation_time_ms, base_dir='../models'):
    # Get the model
    m_data = asdk.get_models_dir(base_dir=base_dir, cell_id=cell_id, model_type=asdk.LIF_ASC)
    if len(m_data) == 0:
        print("Couldn't find a {} model for cell {} in {}.".format(asdk.LIF, cell_id, base_dir))
        return

    if len(m_data) > 1:
        print('Found more than one {} model for cell {}.'.format(asdk.LIF, cell_id))
        return

    # By default NEST has a 0.1 ms resolution which is the which can causes integration issues due to glif_lif_asc
    # using explicit euler method
    # Note: If NEST is called at some previous point you will need to call nest.ResetKernel
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution': dt_ms})

    # Set parameters using config json file
    model = m_data[0]
    config = json_utilities.read(model['model-config-file'])
    coeffs = config['coeffs']
    neuron = nest.Create('glif_lif_asc',
                         params={'V_th': coeffs['th_inf'] * config['th_inf'],
                                 'g': coeffs['G'] / config['R_input'],
                                 'E_L': config['El'],
                                 'C_m': coeffs['C'] * config['C'],
                                 't_ref': config['spike_cut_length'] * dt_ms ,
                                 'asc_init': config['init_AScurrents'],
                                 'k': 1.0 / np.array(config['asc_tau_array']),
                                 'asc_amps': np.array(config['asc_amp_array']) *
                                             np.array(coeffs['asc_amp_array'])})


    # Multimeter to read ASCurrents, Voltmeter and spike reader
    multimeter = nest.Create("multimeter", params={'record_from': ['V_m', 'AScurrents_sum'],
                                                   'withgid': True, 'withtime': True})
    voltmeter = nest.Create("voltmeter", params= {"withgid": True, "withtime": True})
    spikedetector = nest.Create("spike_detector", params={"withgid": True, "withtime": True})
    nest.Connect(multimeter, neuron)
    nest.Connect(voltmeter, neuron)
    nest.Connect(neuron, spikedetector)

    # Step current
    scg = nest.Create("step_current_generator", params={'amplitude_times': amp_times, 'amplitude_values': amp_vals})
    nest.Connect(scg, neuron)

    # Simulate, grab run values and return
    nest.Simulate(simulation_time_ms)
    voltages = nest.GetStatus(voltmeter)[0]['events']['V_m']
    times = nest.GetStatus(voltmeter)[0]['events']['times']
    spike_times = nest.GetStatus(spikedetector)[0]['events']['times']
    return times, voltages, spike_times


def run_long_square(cell_id, pulse_time, amplitude, total_time=1000.0, dt=0.05):
    ret = {}

    amp_times = [0.0, pulse_time[0], pulse_time[1]]
    amp_values = [0.0, amplitude, 0.0]
    output = run_lif_asc_nest(cell_id, amp_times, amp_values, dt, total_time)
    ret['nest'] = {'times': output[0], 'voltages': output[1], 'spike_times': output[2]}

    I = get_step_trace(amp_times, amp_values, dt, total_time)
    output = run_lif_asc_allensdk(cell_id, asdk.LIF_ASC, I, dt)
    ret['allen'] = {'times': output[0], 'voltages': output[1], 'spike_times': output[2]}
    ret['I'] = I

    return ret


def run_short_squares(cell_id, pulses, total_time=1000.0, dt=0.05):
    ret = {}

    amp_times = [0.0]
    amp_values = [0.0]
    for p in pulses:
        amp_times += [p[0], p[0] + p[1]]
        amp_values += [p[2], 0.0]

    output = run_lif_asc_nest(cell_id, amp_times, amp_values, dt, total_time)
    ret['nest'] = {'times': output[0], 'voltages': output[1], 'spike_times': output[2]}

    I = get_step_trace(amp_times, amp_values, dt, total_time)
    output = run_lif_asc_allensdk(cell_id, asdk.LIF_ASC, I, dt)
    ret['allen'] = {'times': output[0], 'voltages': output[1], 'spike_times': output[2]}
    ret['I'] = I

    return ret


def run_ramp(cell_id, max_amp, total_time=1000.0, dt=0.05):
    ret = {}

    n_steps = int(total_time / dt)
    dI_dt = max_amp / total_time
    amp_times = [t*dt for t in xrange(n_steps)]
    amp_values = [t*dI_dt for t in amp_times]
    output = run_lif_asc_nest(cell_id, amp_times, amp_values, dt, total_time)
    ret['nest'] = {'times': output[0], 'voltages': output[1], 'spike_times': output[2]}

    output = run_lif_asc_allensdk(cell_id, asdk.LIF_ASC, amp_values, dt)
    ret['allen'] = {'times': output[0], 'voltages': output[1], 'spike_times': output[2]}
    ret['I'] = amp_values

    return ret


def create_stim_fn(fn, **params):
    return partial(fn, **params)

stimulus = {
    'no-input': create_stim_fn(run_long_square, pulse_time=(100.0, 900.0), amplitude=0.0e-10, total_time=1000.0),

    # long square
    'long-square-1': create_stim_fn(run_long_square, pulse_time=(100.0, 900.0), amplitude=1.0e-10),
    'long-square-2': create_stim_fn(run_long_square, pulse_time=(100.0, 900.0), amplitude=2.5e-10),
    'long-square-3': create_stim_fn(run_long_square, pulse_time=(100.0, 900.0), amplitude=5.0e-10),
    'long-square-4': create_stim_fn(run_long_square, pulse_time=(100.0, 900.0), amplitude=10.0e-10),

    # Single Short Square
    'single-short-square-1': create_stim_fn(run_short_squares, pulses=[(100.0, 15.0, 10.0e-10)], total_time=300.0), # No spike
    'single-short-square-2': create_stim_fn(run_short_squares, pulses=[(100.0, 15.0, 20.0e-10)], total_time=300.0), # One spike
    'single-short-square-3': create_stim_fn(run_short_squares, pulses=[(100.0, 15.0, 40.0e-10)], total_time=300.0), # two spikes

    # Three Short Squares
    'triple-short-square-1': create_stim_fn(run_short_squares, pulses=[(200.0, 40.0, 3.0e-10), (500.0, 40.0, 3.0e-10), (800.0, 40.0, 3.0e-10)], total_time=1000.0), # 0
    'triple-short-square-2': create_stim_fn(run_short_squares, pulses=[(200.0, 40.0, 5.0e-10), (500.0, 40.0, 5.0e-10), (800.0, 40.0, 5.0e-10)], total_time=1000.0), # 1 middle spike
    'triple-short-square-3': create_stim_fn(run_short_squares, pulses=[(200.0, 40.0, 8.0e-10), (500.0, 40.0, 8.0e-10), (800.0, 40.0, 8.0e-10)], total_time=1000.0),
    'triple-short-square-4': create_stim_fn(run_short_squares, pulses=[(200.0, 20.0, 10.0e-10), (500.0, 20.0, 10.0e-10), (800.0, 20.0, 10.0e-10)], total_time=1000.0), # 2 spikes
    'triple-short-square-5': create_stim_fn(run_short_squares, pulses=[(200.0, 20.0, 20.0e-10), (500.0, 20.0, 20.0e-10), (800.0, 20.0, 20.0e-10)], total_time=1000.0), # 2 spikes

    # Ramp
    'ramp-1': create_stim_fn(run_ramp, max_amp=1.0e-10),
    'ramp-2': create_stim_fn(run_ramp, max_amp=2.5e-10),
    'ramp-3': create_stim_fn(run_ramp, max_amp=5e-10),
    'ramp-4': create_stim_fn(run_ramp, max_amp=10.0e-10),
    'ramp-5': create_stim_fn(run_ramp, max_amp=25.0e-10),
    'ramp-6': create_stim_fn(run_ramp, max_amp=10.0e-10, total_time=2000.0),
    'ramp-7': create_stim_fn(run_ramp, max_amp=10.0e-10, total_time=3000.0)
}



# Long Square
#output = run_long_square(323834998, (100.0, 900.0), 0.0e-10, 1000.0, 0.05)
#output = run_long_square(323834998, (100.0, 900.0), 1.0e-10, 1000.0, 0.05) # No spiking
#output = run_long_square(323834998, (100.0, 900.0), 2.5e-10, 1000.0, 0.05)
#output = run_long_square(323834998, (100.0, 900.0), 5.0e-10, 1000.0, 0.05)
#output = run_long_square(323834998, (100.0, 900.0), 10.0e-10, 1000.0, 0.05)

# Single Short Square
#output = run_short_squares(323834998, [(100.0, 15.0, 10.0e-10)], 300.0, 0.05) # No spike
#output = run_short_squares(323834998, [(100.0, 15.0, 20.0e-10)], 300.0, 0.05) # One spike
#output = run_short_squares(323834998, [(100.0, 15.0, 40.0e-10)], 300.0, 0.05) # two spikes

# Three Short Squares

#output = run_short_squares(323834998, [(200.0, 40.0, 3.0e-10), (500.0, 40.0, 3.0e-10), (800.0, 40.0, 3.0e-10)], 1000.0, 0.05) # 0
#output = run_short_squares(323834998, [(200.0, 40.0, 5.0e-10), (500.0, 40.0, 5.0e-10), (800.0, 40.0, 5.0e-10)], 1000.0, 0.05) # 1 middle spike
#output = run_short_squares(323834998, [(200.0, 40.0, 8.0e-10), (500.0, 40.0, 8.0e-10), (800.0, 40.0, 8.0e-10)], 1000.0, 0.05)
#output = run_short_squares(323834998, [(200.0, 20.0, 10.0e-10), (500.0, 20.0, 10.0e-10), (800.0, 20.0, 10.0e-10)], 1000.0, 0.05) # 2 spikes
#output = run_short_squares(323834998, [(200.0, 20.0, 20.0e-10), (500.0, 20.0, 20.0e-10), (800.0, 20.0, 20.0e-10)], 1000.0, 0.05) # 2 spikes


# Ramp
#output = run_ramp(323834998, 1.0e-10, 1000.0, 0.05)
#output = run_ramp(323834998, 2.5e-10, 1000.0, 0.05)
#output = run_ramp(323834998, 5e-10, 1000.0, 0.05)
#output = run_ramp(323834998, 10.0e-10, 1000.0, 0.05)
#output = run_ramp(323834998, 25.0e-10, 1000.0, 0.05)
#output = run_ramp(323834998, 10.0e-10, 2000.0, 0.05)
output = run_ramp(323834998, 10.0e-10, 3000.0, 0.05)
#output = run_ramp(323834998, 5.0e-10, 2000.0, 0.05)



#o = stimulus['triple-short-square-5']
#o = partial(run_ramp, max_amp=2.5e-10, total_time=1000.0, dt=0.05)
#output = o(323834998)

print len(output['I'])
plt_comparison(output['I'],
               output['allen']['times'], output['allen']['voltages'], output['allen']['spike_times'],
               output['nest']['times'], output['nest']['voltages'], output['nest']['spike_times'], show=True)




#amp_times = [0.0, 100.0, 900.0]
#amp_values = [0.0, 2.6e-10, 0.0]
#dt_ms = 0.05
#simulation_time_ms = 1000.0
#times, voltages, spike_times = run_lif_asc_nest(323834998, amp_times, amp_values, dt_ms, simulation_time_ms)
#I = get_step_trace(amp_times, amp_values, times[1]-times[0], (times[1]-times[0])*len(times))
#plt.figure()
#plot_vt(times, voltages, I, show=False)

#I_a = get_step_trace(amp_times, amp_values, dt_ms, simulation_time_ms, 0.0)
#times_a, voltages_a, spike_times_a = run_lif_asc_allensdk(323834998, asdk.LIF_ASC, I_a, dt_ms)
#allen_ts, allen_v, allen_spikes_times = run_lif_asc_allensdk(323834998, asdk.LIF_ASC, I_a, dt_ms)

#allen_spikes_times = allen_spikes_times * 1.0e04

#plt_comparison(I_a, allen_ts, allen_v, times, voltages, allen_spikes_times, spike_times, show=True)

