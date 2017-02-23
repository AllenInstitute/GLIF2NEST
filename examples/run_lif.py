import nest
import sys
import os

import matplotlib.pyplot as plt

from allensdk.model.glif.glif_neuron import GlifNeuron
import allensdk.core.json_utilities as json_utilities

sys.path.append(os.path.join(sys.path[0], '../utils'))
from plot_helper import plot_vt, step_generator_trace, get_step_trace
import allensdk_helper as asdk

#nest.Install('/home/kael/apps/nest-simulator/lib/nest/glifmodule')
nest.Install('glifmodule.so')


def run_lif_allensdk(cell_id, model_type, I, dt_ms, base_dir='../models'):
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

    #print neuron_config['spike_cut_length']*neuron.dt
    #exit()

    output = neuron.run(I)
    voltages = output['voltage']
    times = [t*dt_ms for t in xrange(len(I))]
    spike_times = output['interpolated_spike_times']

    return times, voltages, spike_times





def run_lif_nest(cell_id, amp_times, amp_vals, dt_ms, simulation_time_ms, base_dir='../models'):
    m_data = asdk.get_models_dir(base_dir=base_dir, cell_id=cell_id, model_type=asdk.LIF)

    if len(m_data) == 0:
        print("Couldn't find a {} model for cell {} in {}.".format(asdk.LIF, cell_id, base_dir))
        return

    if len(m_data) > 1:
        print('Found more than one {} model for cell {}.'.format(asdk.LIF, cell_id))
        return

    nest.SetKernelStatus({'resolution': dt_ms})

    model = m_data[0]
    neuron = nest.Create("glif_lif")

    # Set parameters using config json file
    config = json_utilities.read(model['model-config-file'])
    threshold = config['coeffs']['th_inf'] * config['th_inf']
    G = config['coeffs']['G'] / config['R_input']
    El = config['El']
    C = config['coeffs']['C'] * config['C']
    t_ref = config['spike_cut_length'] * dt_ms #config['dt']

    nest.SetStatus(neuron, {'V_th': threshold})
    nest.SetStatus(neuron, {'g': G})
    nest.SetStatus(neuron, {'E_L': El})
    nest.SetStatus(neuron, {'C_m': C})
    nest.SetStatus(neuron, {'t_ref': t_ref})

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

    nest.Simulate(simulation_time_ms)
    voltages = nest.GetStatus(voltmeter)[0]['events']['V_m']
    times = nest.GetStatus(voltmeter)[0]['events']['times']
    spike_times = nest.GetStatus(spikedetector)[0]['events']['times']

    #print nest.GetStatus(spikedetector)[0]['events']
    #exit()

    return times, voltages, spike_times

    #(amp_ts, amp_is) = nest.GetStatus(scg, ['amplitude_times', 'amplitude_values'])[0]
    #I = step_generator_trace(amp_ts, amp_is, times, 0.0)
    #plot_vt(times, voltages, I)

#I_generator = nest.Create("step_current_generator")
#nest.SetStatus(I_generator, {'amplitude_times': amp_times,
#                             'amplitude_values': amp_vals})

amp_times = [0.0, 100.0, 900.0]
amp_values = [0.0, 2.6e-10, 0.0]
#amp_values = [0.0, 2.6e-10, 0.0]
dt_ms = 0.05
simulation_time_ms = 1000.0
#print simulation_time_ms / dt_ms
#exit()

times, voltages, spike_times = run_lif_nest(323834998, amp_times, amp_values, dt_ms, simulation_time_ms)
I = get_step_trace(amp_times, amp_values, times[1]-times[0],
                   (times[1]-times[0])*len(times))

plt.figure(1)
plot_vt(times, voltages, I, show=False)


I_a = get_step_trace(amp_times, amp_values, dt_ms, simulation_time_ms, 0.0)
times_a, voltages_a, spike_times_a = run_lif_allensdk(323834998, asdk.LIF, I_a, dt_ms)

#assert(len(spike_times) == len(spike_times_a))

#print voltages
plt.figure(2)
plot_vt(times_a, voltages_a, I_a, show=False)
plt.show()

#print spike_times*1.0e-04
#print spike_times_a

#nest.ResetKernel()
#nest.SetKernelStatus({'resolution': 0.05})

#neuron = nest.Create("glif_lif")
#print nest.GetStatus(neuron)

#nest.SetStatus(neuron, {'V_th': 0.0283406593326})
#nest.SetStatus(neuron, {'g': 4.469424898637e-09})
#nest.SetStatus(neuron, {'E_L': 0.0})
#nest.SetStatus(neuron, {'C_m': 7.78068598975e-11})
#nest.SetStatus(neuron, {'t_ref': 20.0})
#exit()


#voltmeter = nest.Create("voltmeter")
#nest.SetStatus(voltmeter, {"withgid": True, "withtime": True})

#nest.Connect(voltmeter, neuron)

#scg = nest.Create("step_current_generator")
#nest.SetStatus(scg, {'amplitude_times': [0.0, 100.0, 900.0],
#                     'amplitude_values': [0.0, 2.0e-10, 0.0]})
#nest.Connect(scg, neuron)

#nest.Simulate(1000.0)
#voltages = nest.GetStatus(voltmeter)[0]['events']['V_m']
#times = nest.GetStatus(voltmeter)[0]['events']['times']

#(amp_ts, amp_is) = nest.GetStatus(scg, ['amplitude_times', 'amplitude_values'])[0]
#I = step_generator_trace(amp_ts, amp_is, times, 0.0)
#plot_vt(times, voltages, I)

