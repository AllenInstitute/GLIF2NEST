"""
A helper for running AllenSDK and NEST Glif models, including setting up various types of input
currents. Plots a voltage and spike train comparsion of
ex:
Run a long-square current on the LIF model for cell 52746013
    $ python run_model.py -c 52746013 -m LIF -s long-square-2

Run a short-square injection on three LIF-ASC models
    $ python run_model.py -c 52746013,490205998, -m LIF-ASC -s short-square-3

Run 4 different ramp injections of a LIF model
    $ python run_model.py -c 52746013 -m LIF-R -s ramp-1,ramp-2,ramp-3,ramp-4

Note that the model configuration files need to be downloaded, which can be done using
the --download option or first running allensdk_helper.py
"""


from functools import partial
from optparse import OptionParser
import numpy as np
import matplotlib.pyplot as plt

import nest

from allensdk.model.glif.glif_neuron import GlifNeuron
import allensdk.core.json_utilities as json_utilities
import allensdk_helper as asdk
import plot_helper as plotter

nest.Install('glifmodule.so')


def runGlifNeuron(cell_id, model_type, I, dt_ms, base_dir='../models'):
    """Run a allensdk GlifNeuron and return voltages and spike-times"""
    # Get the neuron configuration file
    m_data = asdk.get_models_dir(base_dir=base_dir, cell_id=cell_id, model_type=model_type)
    if len(m_data) == 0:
        msg = "Couldn't find a {} model for cell {} in {}.".format(model_type, cell_id, base_dir)
        raise Exception(msg)

    if len(m_data) > 1:
        msg = 'Found more than one {} model for cell {}.'.format(model_type, cell_id)
        raise Exception(msg)

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



def create_lif(config_file, dt_ms):
    """Creates a nest glif_lif object"""
    config = json_utilities.read(config_file)
    coeffs = config['coeffs']
    return nest.Create('glif_lif',
                       params={'V_th': coeffs['th_inf'] * config['th_inf'],
                               'g': coeffs['G'] / config['R_input'],
                               'E_L': config['El'],
                               'C_m': coeffs['C'] * config['C'],
                               't_ref': config['spike_cut_length'] * dt_ms})


def create_lif_asc(config_file, dt_ms):
    """Creates a nest glif_lif_asc object"""
    config = json_utilities.read(config_file)
    coeffs = config['coeffs']
    return nest.Create('glif_lif_asc',
                       params={'V_th': coeffs['th_inf'] * config['th_inf'],
                               'g': coeffs['G'] / config['R_input'],
                               'E_L': config['El'],
                               'C_m': coeffs['C'] * config['C'],
                               't_ref': config['spike_cut_length'] * dt_ms,
                               'asc_init': config['init_AScurrents'],
                               'k': 1.0 / np.array(config['asc_tau_array']),
                               'asc_amps': np.array(config['asc_amp_array']) *
                                           np.array(coeffs['asc_amp_array'])})


def runNestModel(cell_id, model_type, amp_times, amp_vals, dt_ms, simulation_time_ms, base_dir='../models'):
    """Creates and runs a NEST glif object and returns the voltages and spike-times"""
    # Get the model
    m_data = asdk.get_models_dir(base_dir=base_dir, cell_id=cell_id, model_type=model_type)
    if len(m_data) == 0:
        msg = "Couldn't find a {} model for cell {} in {}.".format(asdk.LIF, cell_id, base_dir)
        raise Exception(msg)

    if len(m_data) > 1:
        msg = 'Found more than one {} model for cell {}.'.format(asdk.LIF, cell_id)
        raise Exception(msg)

    # By default NEST has a 0.1 ms resolution which is the which can causes integration issues due to glif_lif_asc
    # using explicit euler method
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution': dt_ms})

    # Set parameters using config json file
    model = m_data[0]
    if model_type == asdk.LIF:
        neuron = create_lif(model['model-config-file'], dt_ms)
    elif model_type == asdk.LIF_ASC:
        neuron = create_lif_asc(model['model-config-file'], dt_ms)
        multimeter = nest.Create("multimeter", params={'record_from': ['AScurrents_sum'], 'withgid': True, 'withtime': True})
        nest.Connect(multimeter, neuron)

    # Create voltmeter and spike reader
    voltmeter = nest.Create("voltmeter", params= {"withgid": True, "withtime": True})
    spikedetector = nest.Create("spike_detector", params={"withgid": True, "withtime": True})
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


def run_long_square(cell_id, model_type, pulse_time, amplitude, total_time=1000.0, dt=0.05):
    """Generates a single long constant injection current

    Parameters
    ----------
    cell_id : ID of cell speciment
    model_type : LIF, LIF-R, LIF-R-ASC, LIF-ASC, LIF-R-ASC-A
    pulse_time : A tuple 0 <= (start_time, end_time) <= total_time, in ms
    ampltiude : Amps of input current
    """
    ret = {}

    amp_times = [0.0, pulse_time[0], pulse_time[1]]
    amp_values = [0.0, amplitude, 0.0]
    output = runNestModel(cell_id, model_type, amp_times, amp_values, dt, total_time)
    ret['nest'] = {'times': output[0], 'voltages': output[1], 'spike_times': output[2]}

    I = plotter.get_step_trace(amp_times, amp_values, dt, total_time)
    output = runGlifNeuron(cell_id, model_type, I, dt)
    ret['allen'] = {'times': output[0], 'voltages': output[1], 'spike_times': output[2]}
    ret['I'] = I

    return ret


def run_short_squares(cell_id, model_type, pulses, total_time=1000.0, dt=0.05):
    """Generates a series of square injection currents

    Parameters
    ----------
    cell_id : ID of cell speciment
    model_type : LIF, LIF-R, LIF-R-ASC, LIF-ASC, LIF-R-ASC-A
    pulse_time : A list of tuples (start_time (ms), end_time (ms), amplitude (Amps))
    """
    ret = {}

    amp_times = [0.0]
    amp_values = [0.0]
    for p in pulses:
        amp_times += [p[0], p[0] + p[1]]
        amp_values += [p[2], 0.0]

    output = runNestModel(cell_id, model_type, amp_times, amp_values, dt, total_time)
    ret['nest'] = {'times': output[0], 'voltages': output[1], 'spike_times': output[2]}

    I = plotter.get_step_trace(amp_times, amp_values, dt, total_time)
    output = runGlifNeuron(cell_id, model_type, I, dt)
    ret['allen'] = {'times': output[0], 'voltages': output[1], 'spike_times': output[2]}
    ret['I'] = I

    return ret


def run_ramp(cell_id, model_type, max_amp, total_time=1000.0, dt=0.05):
    """Generates a ramped injection current starting at time 0 and going to total_time

    Parameters
    ----------
    cell_id : ID of cell speciment
    model_type : LIF, LIF-R, LIF-R-ASC, LIF-ASC, LIF-R-ASC-A
    max_amp : maximum Amp value (occurs at total_time)
    """
    ret = {}

    n_steps = int(total_time / dt)
    dI_dt = max_amp / total_time
    amp_times = [t*dt for t in xrange(n_steps)]
    amp_values = [t*dI_dt for t in amp_times]
    output = runNestModel(cell_id, model_type, amp_times, amp_values, dt, total_time)
    ret['nest'] = {'times': output[0], 'voltages': output[1], 'spike_times': output[2]}

    output = runGlifNeuron(cell_id, model_type, amp_values, dt)
    ret['allen'] = {'times': output[0], 'voltages': output[1], 'spike_times': output[2]}
    ret['I'] = amp_values

    return ret


def create_stim_fn(fn, **params):
    return partial(fn, **params)


stimulus = {
    'no-input': create_stim_fn(run_long_square, pulse_time=(100.0, 900.0), amplitude=0.0e-10, total_time=1000.0),

    # long square
    'long-square-1': create_stim_fn(run_long_square, pulse_time=(100.0, 900.0), amplitude=1.0e-10),
    'long-square-2': create_stim_fn(run_long_square, pulse_time=(100.0, 900.0), amplitude=3.5e-10),
    'long-square-3': create_stim_fn(run_long_square, pulse_time=(100.0, 900.0), amplitude=10.0e-10),
    'long-square-4': create_stim_fn(run_long_square, pulse_time=(100.0, 900.0), amplitude=15.0e-10),

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


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-d", "--dir", dest="base_dir", default="../models", help="location of model directories.")
    parser.add_option("--download", action="store_true", dest="download", default=False, help="download model if not availble locally.")
    parser.add_option("-c", "--cells", dest="cells", default="", help="comma separated list of cells to run.")
    parser.add_option("-m", "--model", dest="model", default="LIF", help="GLIF model type to run.")
    parser.add_option("-s", "--stimulus", dest="stimulus", default="long-square-2", help="Name of current injection to run on cell.")
    parser.add_option("--list-stimuli", action="store_true", dest="list_stim", default=False, help="List all available current stimuli options.")
    options, args = parser.parse_args()

    if options.list_stim:
        print(stimulus.keys())
        exit()

    cell_ids = options.cells.split(',')
    if len(cell_ids) == 0:
        print('No cells specified, please use -c option.')
        exit(1)

    if options.download:
        asdk.download_glif_models(cell_ids, options.base_dir)

    for cell_id in cell_ids:
        for stim in options.stimulus.split(','):
            simulate = stimulus[stim]
            output = simulate(cell_id, options.model)
            plt.figure(stim)
            plotter.plt_comparison(output['I'],
                                   output['allen']['times'], output['allen']['voltages'], output['allen']['spike_times'],
                                   output['nest']['times'], output['nest']['voltages'], output['nest']['spike_times'], show=False)
    plt.show()
