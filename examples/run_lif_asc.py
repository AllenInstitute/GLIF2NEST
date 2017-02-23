import os
import sys
import nest

import numpy as np
import matplotlib.pyplot as plt

from allensdk.model.glif.glif_neuron import GlifNeuron
import allensdk.core.json_utilities as json_utilities

sys.path.append(os.path.join(sys.path[0], '../utils'))
from plot_helper import plot_vt, step_generator_trace, get_step_trace
import allensdk_helper as asdk


nest.Install('glifmodule.so')

#neuron = nest.Create('glif_lif_asc')
#print nest.GetStatus(neuron, 'asc_init')
#nest.SetStatus(neuron, {'asc_init': [1.0, 2.0]})
#print nest.GetStatus(neuron, 'asc_init')

def run_lif_asc_allensdk(cell_id, model_type, I, dt_ms, base_dir='../models'):
    m_data = asdk.get_models_dir(base_dir=base_dir, cell_id=cell_id, model_type=model_type)

    if len(m_data) == 0:
        print("Couldn't find a {} model for cell {} in {}.".format(model_type, cell_id, base_dir))
        return

    if len(m_data) > 1:
        print('Found more than one {} model for cell {}.'.format(model_type, cell_id))
        return

    model_files = m_data[0]
    model_config_file = model_files['model-config-file']
    neuron_config = json_utilities.read(model_config_file)
    neuron = GlifNeuron.from_dict(neuron_config)
    neuron.dt = dt_ms * 1.0e-04

    print("----GlifNeuron----")
    print("coeffs[th_inf] = {}".format(neuron.coeffs['th_inf']))
    print("th_inf = {}".format(neuron.th_inf))
    print("coeff[th_inf] * th_inf = {}".format(neuron.coeffs['th_inf']*neuron.th_inf))
    print("threshold = {}".format(neuron.coeffs['th_inf']*neuron.th_inf))

    #print("init_AScurrents = {}".format(neuron.init_AScurrents))
    #print("k = {}".format(neuron.k))
    #print("dt*cut_length = {}".format(neuron.dt*neuron.spike_cut_length))
    #print("asc_amps * coef[asc_amps] = {} * {} = {}".format(neuron.asc_amp_array, neuron.coeffs['asc_amp_array'], neuron.asc_amp_array*neuron.coeffs['asc_amp_array']))
    #print("r = {}".format(neuron.r))

    #print neuron_config['spike_cut_length']*neuron.dt
    #print(np.exp(-neuron.k[0]*neuron.dt))
    #print(np.exp(-neuron.k[1]*neuron.dt))
    #print(np.exp(-neuron.k*neuron.dt))
    
    output = neuron.run(I)
    voltages = output['voltage']
    times = [t*dt_ms for t in xrange(len(I))]
    spike_times = output['interpolated_spike_times']

    return times, voltages, spike_times


def run_lif_asc_nest(cell_id, amp_times, amp_vals, dt_ms, simulation_time_ms, base_dir='../models'):
    m_data = asdk.get_models_dir(base_dir=base_dir, cell_id=cell_id, model_type=asdk.LIF_ASC)

    if len(m_data) == 0:
        print("Couldn't find a {} model for cell {} in {}.".format(asdk.LIF, cell_id, base_dir))
        return

    if len(m_data) > 1:
        print('Found more than one {} model for cell {}.'.format(asdk.LIF, cell_id))
        return

    nest.SetKernelStatus({'resolution': dt_ms})

    model = m_data[0]
    neuron = nest.Create("glif_lif_asc")

    # Set parameters using config json file
    config = json_utilities.read(model['model-config-file'])
    threshold = config['coeffs']['th_inf'] * config['th_inf']
    G = config['coeffs']['G'] / config['R_input']
    El = config['El']
    C = config['coeffs']['C'] * config['C']
    t_ref = config['spike_cut_length'] * dt_ms 
    
    init_AScurrents = config['init_AScurrents']
    k = 1.0 / np.array(config['asc_tau_array'])
    asc_amps = np.array(config['asc_amp_array']) * np.array(config['coeffs']['asc_amp_array'])

	#print("asc_amps * coef[asc_amps] = {} * {} = {}".format(neuron.asc_amp_array, neuron.coeffs['asc_amp_array'], neuron.asc_amp_array*neuron.coeffs['asc_amp_array']))
    

    nest.SetStatus(neuron, {'V_th': threshold})
    nest.SetStatus(neuron, {'g': G})
    nest.SetStatus(neuron, {'E_L': El})
    nest.SetStatus(neuron, {'C_m': C})
    nest.SetStatus(neuron, {'t_ref': t_ref})
    
    nest.SetStatus(neuron, {'asc_init': init_AScurrents})
    nest.SetStatus(neuron, {'k': k})
    nest.SetStatus(neuron, {'asc_amps': asc_amps})

    voltmeter = nest.Create("voltmeter")
    spikedetector = nest.Create("spike_detector",
                                params={"withgid": True, "withtime": True})
    nest.SetStatus(voltmeter, {"withgid": True, "withtime": True})


    nest.Connect(voltmeter, neuron)
    nest.Connect(neuron, spikedetector)

    scg = nest.Create("step_current_generator")
    nest.SetStatus(scg, {'amplitude_times': amp_times,
                         'amplitude_values': amp_vals})
    nest.Connect(scg, neuron)

    print('******Nest******')
    #threshold = config['coeffs']['th_inf'] * config['th_inf']
    print("coeffs[th_inf] = " + str(config['coeffs']['th_inf']))
    print("th_inf = {}".format(config['th_inf']))
    print("coeff[th_inf] * th_inf = {}".format(config['coeffs']['th_inf']*config['th_inf']))
    print("threshold = {}".format(nest.GetStatus(neuron, 'V_th')[0]))

    nest.Simulate(simulation_time_ms)
    voltages = nest.GetStatus(voltmeter)[0]['events']['V_m']
    times = nest.GetStatus(voltmeter)[0]['events']['times']
    spike_times = nest.GetStatus(spikedetector)[0]['events']['times']





    #print("init_AScurrents = {}".format(nest.GetStatus(neuron, 'asc_init')[0]))
    #print("k = {}".format(nest.GetStatus(neuron, 'k')[0]))
    #print("dt*cut_length = {}".format(nest.GetStatus(neuron, 't_ref')[0]))
    #print("asc_amps = {}".format(nest.GetStatus(neuron, 'asc_amps')[0]))

    #print nest.GetStatus(spikedetector)[0]['events']
    #exit()

    return times, voltages, spike_times





amp_times = [0.0, 100.0, 900.0]
amp_values = [0.0, 2.6e-10, 0.0]
dt_ms = 0.05
simulation_time_ms = 1000.0
#
times, voltages, spike_times = run_lif_asc_nest(323834998, amp_times, amp_values, dt_ms, simulation_time_ms)
I = get_step_trace(amp_times, amp_values, times[1]-times[0],
                   (times[1]-times[0])*len(times))
plt.figure()
plot_vt(times, voltages, I, show=False)

I_a = get_step_trace(amp_times, amp_values, dt_ms, simulation_time_ms, 0.0)
times_a, voltages_a, spike_times_a = run_lif_asc_allensdk(323834998, asdk.LIF_ASC, I_a, dt_ms)

plt.figure()
plot_vt(times_a, voltages_a, I_a, show=False)
plt.show()