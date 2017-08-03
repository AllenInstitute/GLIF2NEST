"""
A helper for running NEST Glif models with/without current-based synaptic ports, 
including setting up various types of input currents. 
Plots a voltage and spike train comparsion of neurons
ex:
Run a long-square current on the LIF model for cell 52746013
    $ python run_model_psc.py -c 52746013 -m LIF -s long-square-2

Run a short-square injection on three LIF-ASC models
    $ python run_model_psc.py -c 52746013,490205998, -m LIF-ASC -s short-square-3

Run 4 different ramp injections of a LIF model
    $ python run_model_psc.py -c 52746013 -m LIF-R -s ramp-1,ramp-2,ramp-3,ramp-4
"""

from functools import partial
from optparse import OptionParser
import numpy as np
import matplotlib.pyplot as plt

import nest

from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.glif_api import GlifApi
import allensdk_helper as asdk
import plot_helper as plotter

nest.Install('glifmodule.so')

def create_lif(config, dt_ms):
    """Creates a nest glif_lif object"""
    coeffs = config['coeffs']
    return nest.Create('glif_lif',
                       params={'V_th': coeffs['th_inf'] * config['th_inf'] * 1.0e03 + config['El_reference'] * 1.0e03,
                               'g': coeffs['G'] / config['R_input'] * 1.0e09,
                               'E_L': config['El']  * 1.0e03 + config['El_reference'] * 1.0e03,
                               'C_m': coeffs['C'] * config['C'] * 1.0e12,
                               't_ref': config['spike_cut_length'] * dt_ms,
                               'V_reset': config['El_reference'] * 1.0e03,
                               #'V_dynamics_method': config['voltage_dynamics_method']['name']}) #'linear_forward_euler' or 'linear_exact'
                               'V_dynamics_method': 'linear_exact'}) #'linear_forward_euler' or 'linear_exact'

def create_lif_asc(config, dt_ms):
    """Creates a nest glif_lif_asc object"""
    coeffs = config['coeffs']
    return nest.Create('glif_lif_asc',
                       params={'V_th': coeffs['th_inf'] * config['th_inf'] * 1.0e03 + config['El_reference'] * 1.0e03,
                               'g': coeffs['G'] / config['R_input'] * 1.0e09,
                               'E_L': config['El'] * 1.0e03 + config['El_reference'] * 1.0e03,
                               'C_m': coeffs['C'] * config['C'] * 1.0e12,
                               't_ref': config['spike_cut_length'] * dt_ms,
                               'V_reset': config['El_reference'] * 1.0e03,
                               'asc_init': np.array(config['init_AScurrents']) * 1.0e12,
                               'k': 1.0 / np.array(config['asc_tau_array']) * 1.0e-03,
                               'asc_amps': np.array(config['asc_amp_array']) *
                                           np.array(coeffs['asc_amp_array']) * 1.0e12,
                               #'V_dynamics_method': config['voltage_dynamics_method']['name']}) #'linear_forward_euler' or 'linear_exact'
                               'V_dynamics_method': 'linear_exact'})
    
def create_lif_r(config, dt_ms):
    """Creates a nest glif_lif_r object"""
    coeffs = config['coeffs']
    threshold_params = config['threshold_dynamics_method']['params']
    reset_params = config['voltage_reset_method']['params']
    return nest.Create('glif_lif_r',
                       params={'V_th': coeffs['th_inf'] * config['th_inf'] * 1.0e03 + config['El_reference'] * 1.0e03,
                               'g': coeffs['G'] / config['R_input'] * 1.0e09,
                               'E_L': config['El'] * 1.0e03 + config['El_reference'] * 1.0e03,
                               'C_m': coeffs['C'] * config['C'] * 1.0e12,
                               't_ref': config['spike_cut_length'] * dt_ms,
                               #'V_reset': config['El_reference'] * 1.0e03,
                               'a_spike': threshold_params['a_spike'] * 1.0e03,
                               'b_spike': threshold_params['b_spike'] * 1.0e-03,
                               'a_reset': reset_params['a'], 
                               'b_reset': reset_params['b'] * 1.0e03,
                               #'V_dynamics_method': config['voltage_dynamics_method']['name']}) #'linear_forward_euler' or 'linear_exact'
                               'V_dynamics_method': 'linear_exact'})  
    
def create_lif_r_asc(config, dt_ms):
    """Creates a nest glif_lif_r_asc object"""
    coeffs = config['coeffs']
    threshold_params = config['threshold_dynamics_method']['params']
    reset_params = config['voltage_reset_method']['params']
    return nest.Create('glif_lif_r_asc',
                       params={'V_th': coeffs['th_inf'] * config['th_inf'] * 1.0e03 + config['El_reference'] * 1.0e03,
                               'g': coeffs['G'] / config['R_input'] * 1.0e09,
                               'E_L': config['El'] * 1.0e03 + config['El_reference'] * 1.0e03,
                               'C_m': coeffs['C'] * config['C'] * 1.0e12,
                               't_ref': config['spike_cut_length'] * dt_ms,
                               'a_spike': threshold_params['a_spike'] * 1.0e03,
                               'b_spike': threshold_params['b_spike'] * 1.0e-03,
                               'a_reset': reset_params['a'], 
                               'b_reset': reset_params['b'] * 1.0e03,
                               'asc_init': np.array(config['init_AScurrents']) * 1.0e12,
                               'k': 1.0 / np.array(config['asc_tau_array']) * 1.0e-03,
                               'asc_amps': np.array(config['asc_amp_array']) *
                                           np.array(coeffs['asc_amp_array']) * 1.0e12,
                               #'V_dynamics_method': config['voltage_dynamics_method']['name']}) #'linear_forward_euler' or 'linear_exact'
                               'V_dynamics_method': 'linear_exact'})

def create_lif_r_asc_a(config, dt_ms):
    """Creates a nest glif_lif_r_asc_a object"""
    coeffs = config['coeffs']
    threshold_params = config['threshold_dynamics_method']['params']
    reset_params = config['voltage_reset_method']['params']
    return nest.Create('glif_lif_r_asc_a',
                       params={'V_th': coeffs['th_inf'] * config['th_inf'] * 1.0e03 + config['El_reference'] * 1.0e03,
                               'g': coeffs['G'] / config['R_input'] * 1.0e09,
                               'E_L': config['El'] * 1.0e03 + config['El_reference'] * 1.0e03,
                               'C_m': coeffs['C'] * config['C'] * 1.0e12,
                               't_ref': config['spike_cut_length'] * dt_ms,
                               'a_spike': threshold_params['a_spike'] * 1.0e03,
                               'b_spike': threshold_params['b_spike'] * 1.0e-03,
                               'a_voltage': threshold_params['a_voltage'] * coeffs['a'] * 1.0e-03,
                               'b_voltage': threshold_params['b_voltage'] * coeffs['b'] * 1.0e-03,
                               'a_reset': reset_params['a'], 
                               'b_reset': reset_params['b'] * 1.0e03,
                               'asc_init': np.array(config['init_AScurrents']) * 1.0e12,
                               'k': 1.0 / np.array(config['asc_tau_array']) * 1.0e-03,
                               'asc_amps': np.array(config['asc_amp_array']) *
                                           np.array(coeffs['asc_amp_array']) * 1.0e12,
                               #'V_dynamics_method': config['voltage_dynamics_method']['name']}) #'linear_forward_euler' or 'linear_exact'
                               'V_dynamics_method': 'linear_exact'})

## synaptic ports testing

def create_lif_psc(config, syn_tau, dt_ms):
    """Creates a nest glif_lif_psc object"""
    coeffs = config['coeffs']
    return nest.Create('glif_lif_psc', 
                       params={'V_th': coeffs['th_inf'] * config['th_inf'] * 1.0e03 + config['El_reference'] * 1.0e03,
                               'g': coeffs['G'] / config['R_input'] * 1.0e09,
                               'E_L': config['El']  * 1.0e03 + config['El_reference'] * 1.0e03,
                               'C_m': coeffs['C'] * config['C'] * 1.0e12,
                               't_ref': config['spike_cut_length'] * dt_ms,
                               'V_reset': config['El_reference'] * 1.0e03,
                               'tau_syn': syn_tau, # in ms
                               #'V_dynamics_method': config['voltage_dynamics_method']['name']}) #'linear_forward_euler' or 'linear_exact'
                               'V_dynamics_method': 'linear_exact'}) #'linear_forward_euler' or 'linear_exact'

def create_lif_r_psc(config, syn_tau, dt_ms):
    """Creates a nest glif_lif_r_psc object"""
    coeffs = config['coeffs']
    threshold_params = config['threshold_dynamics_method']['params']
    reset_params = config['voltage_reset_method']['params']
    return nest.Create('glif_lif_r_psc',
                       params={'V_th': coeffs['th_inf'] * config['th_inf'] * 1.0e03 + config['El_reference'] * 1.0e03,
                               'g': coeffs['G'] / config['R_input'] * 1.0e09,
                               'E_L': config['El'] * 1.0e03 + config['El_reference'] * 1.0e03,
                               'C_m': coeffs['C'] * config['C'] * 1.0e12,
                               't_ref': config['spike_cut_length'] * dt_ms,
                               #'V_reset': config['El_reference'] * 1.0e03,
                               'a_spike': threshold_params['a_spike'] * 1.0e03,
                               'b_spike': threshold_params['b_spike'] * 1.0e-03,
                               'a_reset': reset_params['a'], 
                               'b_reset': reset_params['b'] * 1.0e03,
                               'tau_syn': syn_tau, # in ms
                               #'V_dynamics_method': config['voltage_dynamics_method']['name']}) #'linear_forward_euler' or 'linear_exact'
                               'V_dynamics_method': 'linear_exact'})  

def create_lif_asc_psc(config, syn_tau, dt_ms):
    """Creates a nest glif_lif_asc_psc object"""
    coeffs = config['coeffs']
    return nest.Create('glif_lif_asc_psc',
                       params={'V_th': coeffs['th_inf'] * config['th_inf'] * 1.0e03 + config['El_reference'] * 1.0e03,
                               'g': coeffs['G'] / config['R_input'] * 1.0e09,
                               'E_L': config['El'] * 1.0e03 + config['El_reference'] * 1.0e03,
                               'C_m': coeffs['C'] * config['C'] * 1.0e12,
                               't_ref': config['spike_cut_length'] * dt_ms,
                               'V_reset': config['El_reference'] * 1.0e03,
                               'asc_init': np.array(config['init_AScurrents']) * 1.0e12,
                               'k': 1.0 / np.array(config['asc_tau_array']) * 1.0e-03,
                               'asc_amps': np.array(config['asc_amp_array']) *
                                           np.array(coeffs['asc_amp_array']) * 1.0e12,
                               'tau_syn': syn_tau, # in ms
                               #'V_dynamics_method': config['voltage_dynamics_method']['name']}) #'linear_forward_euler' or 'linear_exact'
                               'V_dynamics_method': 'linear_exact'})

def create_lif_r_asc_psc(config, syn_tau, dt_ms):
    """Creates a nest glif_lif_r_asc_psc object"""
    coeffs = config['coeffs']
    threshold_params = config['threshold_dynamics_method']['params']
    reset_params = config['voltage_reset_method']['params']
    return nest.Create('glif_lif_r_asc_psc',
                       params={'V_th': coeffs['th_inf'] * config['th_inf'] * 1.0e03 + config['El_reference'] * 1.0e03,
                               'g': coeffs['G'] / config['R_input'] * 1.0e09,
                               'E_L': config['El'] * 1.0e03 + config['El_reference'] * 1.0e03,
                               'C_m': coeffs['C'] * config['C'] * 1.0e12,
                               't_ref': config['spike_cut_length'] * dt_ms,
                               'a_spike': threshold_params['a_spike'] * 1.0e03,
                               'b_spike': threshold_params['b_spike'] * 1.0e-03,
                               'a_reset': reset_params['a'], 
                               'b_reset': reset_params['b'] * 1.0e03,
                               'asc_init': np.array(config['init_AScurrents']) * 1.0e12,
                               'k': 1.0 / np.array(config['asc_tau_array']) * 1.0e-03,
                               'asc_amps': np.array(config['asc_amp_array']) *
                                           np.array(coeffs['asc_amp_array']) * 1.0e12,
                               'tau_syn': syn_tau, # in ms
                               #'V_dynamics_method': config['voltage_dynamics_method']['name']}) #'linear_forward_euler' or 'linear_exact'
                               'V_dynamics_method': 'linear_exact'})

def create_lif_r_asc_a_psc(config, syn_tau, dt_ms):
    """Creates a nest glif_lif_r_asc_a_psc object"""
    coeffs = config['coeffs']
    threshold_params = config['threshold_dynamics_method']['params']
    reset_params = config['voltage_reset_method']['params']
    return nest.Create('glif_lif_r_asc_a_psc',
                       params={'V_th': coeffs['th_inf'] * config['th_inf'] * 1.0e03 + config['El_reference'] * 1.0e03,
                               'g': coeffs['G'] / config['R_input'] * 1.0e09,
                               'E_L': config['El'] * 1.0e03 + config['El_reference'] * 1.0e03,
                               'C_m': coeffs['C'] * config['C'] * 1.0e12,
                               't_ref': config['spike_cut_length'] * dt_ms,
                               'a_spike': threshold_params['a_spike'] * 1.0e03,
                               'b_spike': threshold_params['b_spike'] * 1.0e-03,
                               'a_voltage': threshold_params['a_voltage'] * coeffs['a'] * 1.0e-03,
                               'b_voltage': threshold_params['b_voltage'] * coeffs['b'] * 1.0e-03,
                               'a_reset': reset_params['a'], 
                               'b_reset': reset_params['b'] * 1.0e03,
                               'asc_init': np.array(config['init_AScurrents']) * 1.0e12,
                               'k': 1.0 / np.array(config['asc_tau_array']) * 1.0e-03,
                               'asc_amps': np.array(config['asc_amp_array']) *
                                           np.array(coeffs['asc_amp_array']) * 1.0e12,
                               'tau_syn': syn_tau, # in ms
                               #'V_dynamics_method': config['voltage_dynamics_method']['name']}) #'linear_forward_euler' or 'linear_exact'
                               'V_dynamics_method': 'linear_exact'})

def runNestModel(model_type, neuron_config, amp_times, amp_vals, dt_ms, simulation_time_ms):
    """
    Creates and runs four NEST glif neurons and returns the voltages and spike-times
    One neuron is without synaptic port, the other three are with 2 syaptic ports (one port is 2.0ms and one port is 1.0ms)
    The first neuron is connected the first port of the second neuron
    The first neuron is connected the second port of the third neuron
    The first neuron is also connected both ports of the fourth neuron
    The weights between first neuron and other neurons are all 1000.0
    """

    # By default NEST has a 0.1 ms resolution 
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution': dt_ms})
    
    n=4
    w=1000.0
    syn_tau=[2.0,1.0]
    neurons=[]
    if model_type == asdk.LIF:
        neurons.append( create_lif(neuron_config, dt_ms))
        for i in range(n-1):
            neurons.append( create_lif_psc(neuron_config, syn_tau, dt_ms))
    elif model_type == asdk.LIF_ASC:
        neurons.append( create_lif_asc(neuron_config, dt_ms))
        for i in range(n-1):
            neurons.append( create_lif_asc_psc(neuron_config, syn_tau, dt_ms))
    elif model_type == asdk.LIF_R:
        neurons.append( create_lif_r(neuron_config, dt_ms))
        for i in range(n-1):
            neurons.append( create_lif_r_psc(neuron_config, syn_tau, dt_ms))
    elif model_type == asdk.LIF_R_ASC:
        neurons.append( create_lif_r_asc(neuron_config, dt_ms))
        for i in range(n-1):
            neurons.append( create_lif_r_asc_psc(neuron_config, syn_tau, dt_ms))
    elif model_type == asdk.LIF_R_ASC_A:
        neurons.append( create_lif_r_asc_a(neuron_config, dt_ms))
        for i in range(n-1):
            neurons.append( create_lif_r_asc_a_psc(neuron_config, syn_tau, dt_ms))

    # Create voltmeter and spike reader
    voltmeters=[]
    spikedetectors=[]
    for i in range(n):
        voltmeters.append(nest.Create("voltmeter", params= {"withgid": True, "withtime": True,'interval': dt_ms}))
        # grid spike time
        #spikedetectors.append(nest.Create("spike_detector", params={"withgid": True, "withtime": True,  "precise_times": False}))
        # precise spike time
        spikedetectors.append(nest.Create("spike_detector", params={"withgid": True, "withtime": True}))
        
        nest.Connect(voltmeters[i], neurons[i])
        if i!=0 and i<n-1:
            nest.Connect(neurons[0], neurons[i], syn_spec={'delay': dt_ms, 'weight': w, 'receptor_type': i})
        elif i==n-1:
            nest.Connect(neurons[0], neurons[i], syn_spec={'delay': dt_ms, 'weight': w, 'receptor_type': i-2})
            nest.Connect(neurons[0], neurons[i], syn_spec={'delay': dt_ms, 'weight': w, 'receptor_type': i-1})
    
        nest.Connect(neurons[i], spikedetectors[i])

    # Step current for first neuron
    scg = nest.Create("step_current_generator", params={'amplitude_times': amp_times, 'amplitude_values': np.array(amp_vals) * 1.0e12}) # convert to pF
    nest.Connect(scg, neurons[0],syn_spec={'delay': dt_ms})

    # Simulate, grab run values and return
    nest.Simulate(simulation_time_ms)
    
    # output
    voltages=[]
    times=[]
    spike_times=[]
    
    for i in range(n):
        voltages.append(nest.GetStatus(voltmeters[i])[0]['events']['V_m'])    
        times.append(nest.GetStatus(voltmeters[i])[0]['events']['times'])
        spike_times.append(nest.GetStatus(spikedetectors[i])[0]['events']['times'])

    return times, voltages, spike_times

def run_long_square(cell_id, model_type, neuron_config, pulse_time, amplitude, total_time=1000.0, dt=0.005):
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
    output = runNestModel(model_type, neuron_config, amp_times, amp_values, dt, total_time)
    
    ret['times'] = output[0]
    ret['voltages'] = output[1]
    ret['spike_times'] = output[2]

    I = plotter.get_step_trace(amp_times, amp_values, dt, total_time)  
    ret['I'] = I
    ret['dt']=dt
    
    return ret


def run_short_squares(cell_id, model_type, neuron_config, pulses, total_time=1000.0, dt=0.005):
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

    output = runNestModel(model_type, neuron_config, amp_times, amp_values, dt, total_time)
    
    ret['times'] = output[0]
    ret['voltages'] = output[1]
    ret['spike_times'] = output[2]

    I = plotter.get_step_trace(amp_times, amp_values, dt, total_time)
    ret['I'] = I
    ret['dt']=dt
    
    return ret


def run_short_squares_noise(cell_id, model_type, neuron_config, pulses, total_time=1000.0, dt=0.005):
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
        
    I = plotter.get_step_trace_noise(amp_times, amp_values, dt, total_time)
    It=[t*dt for t in xrange(len(I))]
    output = runNestModel(model_type, neuron_config, It, I, dt, total_time)
    
    ret['times'] = output[0]
    ret['voltages'] = output[1]
    ret['spike_times'] = output[2]
    
    ret['I'] = I
    ret['dt']=dt
    
    return ret

def run_ramp(cell_id, model_type, neuron_config, max_amp, total_time=1000.0, dt=0.005):
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
    output = runNestModel(model_type, neuron_config, amp_times, amp_values, dt, total_time)
    
    ret['times'] = output[0]
    ret['voltages'] = output[1]
    ret['spike_times'] = output[2]
    
    ret['I'] = amp_values
    ret['dt']=dt
    
    return ret

def run_nwb(cell_id, model_type, neuron_config, stim_type='Ramp'):
    """Generates a injection current from nwb sweep file

    Parameters
    ----------
    cell_id : ID of cell speciment
    model_type : LIF, LIF-R, LIF-R-ASC, LIF-ASC, LIF-R-ASC-A
    stumilus_data : stumilus data from NWB file
    """

    # get sweep/stimulus data
    ctc = CellTypesCache()
    ephys_sweeps=ctc.get_ephys_sweeps(cell_id)
    ds=ctc.get_ephys_data(cell_id)   
    ephys_sweep_stim = [s for s in ephys_sweeps if s['stimulus_name'] == stim_type ]   
    ephys_sweep=ephys_sweep_stim[0]
                     
    stumilus_data = ds.get_sweep(ephys_sweep['sweep_number'])
    
    ret = {}

    n_steps = len(stumilus_data['stimulus'])
    dt=1.0 / stumilus_data['sampling_rate'] * 1.0e03
    amp_times = [t*dt for t in xrange(n_steps)]
    amp_values = stumilus_data['stimulus'].tolist()
    total_time=n_steps*dt
    
    output = runNestModel(model_type, neuron_config, amp_times, amp_values, dt, total_time)
    ret['times'] = output[0]
    ret['voltages'] = output[1]
    ret['spike_times'] = output[2]

    ret['I'] = amp_values
    ret['dt']=dt
    
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

    # Single Short Square with noise
    'single-short-square-noise-1': create_stim_fn(run_short_squares_noise, pulses=[(100.0, 15.0, 10.0e-10)], total_time=300.0), # No spike
    'single-short-square-noise-2': create_stim_fn(run_short_squares_noise, pulses=[(100.0, 15.0, 20.0e-10)], total_time=300.0), # One spike
    'single-short-square-noise-3': create_stim_fn(run_short_squares_noise, pulses=[(100.0, 15.0, 40.0e-10)], total_time=300.0), # two spikes


    # Three Short Squares
    'triple-short-square-1': create_stim_fn(run_short_squares, pulses=[(200.0, 40.0, 3.0e-10), (500.0, 40.0, 3.0e-10), (800.0, 40.0, 3.0e-10)], total_time=1000.0), # 0
    'triple-short-square-2': create_stim_fn(run_short_squares, pulses=[(200.0, 40.0, 5.0e-10), (500.0, 40.0, 5.0e-10), (800.0, 40.0, 5.0e-10)], total_time=1000.0), # 1 middle spike
    'triple-short-square-3': create_stim_fn(run_short_squares, pulses=[(200.0, 40.0, 8.0e-10), (500.0, 40.0, 8.0e-10), (800.0, 40.0, 8.0e-10)], total_time=1000.0),
    'triple-short-square-4': create_stim_fn(run_short_squares, pulses=[(200.0, 20.0, 10.0e-10), (500.0, 20.0, 10.0e-10), (800.0, 20.0, 10.0e-10)], total_time=1000.0), # 2 spikes
    'triple-short-square-5': create_stim_fn(run_short_squares, pulses=[(200.0, 20.0, 20.0e-10), (500.0, 20.0, 20.0e-10), (800.0, 20.0, 20.0e-10)], total_time=1000.0), # 2 spikes

    # Three Short Squares with noise
    'triple-short-square-noise-1': create_stim_fn(run_short_squares_noise, pulses=[(200.0, 40.0, 3.0e-10), (500.0, 40.0, 3.0e-10), (800.0, 40.0, 3.0e-10)], total_time=1000.0), # 0
    'triple-short-square-noise-2': create_stim_fn(run_short_squares_noise, pulses=[(200.0, 40.0, 5.0e-10), (500.0, 40.0, 5.0e-10), (800.0, 40.0, 5.0e-10)], total_time=1000.0), # 1 middle spike
    'triple-short-square-noise-3': create_stim_fn(run_short_squares_noise, pulses=[(200.0, 40.0, 8.0e-10), (500.0, 40.0, 8.0e-10), (800.0, 40.0, 8.0e-10)], total_time=1000.0),
    'triple-short-square-noise-4': create_stim_fn(run_short_squares_noise, pulses=[(200.0, 20.0, 10.0e-10), (500.0, 20.0, 10.0e-10), (800.0, 20.0, 10.0e-10)], total_time=1000.0), # 2 spikes
    'triple-short-square-noise-5': create_stim_fn(run_short_squares_noise, pulses=[(200.0, 20.0, 20.0e-10), (500.0, 20.0, 20.0e-10), (800.0, 20.0, 20.0e-10)], total_time=1000.0), # 2 spikes

    # Ramp
    'ramp-1': create_stim_fn(run_ramp, max_amp=1.0e-10),
    'ramp-2': create_stim_fn(run_ramp, max_amp=2.5e-10),
    'ramp-3': create_stim_fn(run_ramp, max_amp=5e-10),
    'ramp-4': create_stim_fn(run_ramp, max_amp=10.0e-10),
    'ramp-5': create_stim_fn(run_ramp, max_amp=25.0e-10),
    'ramp-6': create_stim_fn(run_ramp, max_amp=10.0e-10, total_time=2000.0),
    'ramp-7': create_stim_fn(run_ramp, max_amp=10.0e-10, total_time=3000.0),
        
    # Run stimulus from NWB data file
    'nwb-ramp': create_stim_fn(run_nwb, stim_type='Ramp'),
    'nwb-short-square': create_stim_fn(run_nwb, stim_type='Short Square'),
    'nwb-long-square': create_stim_fn(run_nwb, stim_type='Long Square'),
    'nwb-short-square-triple': create_stim_fn(run_nwb, stim_type='Short Square - Triple'),
    'nwb-square-05ms-subthreshold': create_stim_fn(run_nwb, stim_type='Square - 0.5ms Subthreshold'),
    'nwb-square-2s-suprathreshold': create_stim_fn(run_nwb, stim_type='Square - 2s Suprathreshold'),
    'nwb-noise-1': create_stim_fn(run_nwb, stim_type='Noise 1'),
    'nwb-noise-2': create_stim_fn(run_nwb, stim_type='Noise 2'),
    'nwb-test': create_stim_fn(run_nwb, stim_type='Test')
    
}

if __name__ == '__main__':
    parser = OptionParser()
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

    # model name and model template id mapping
    LIF = 'LIF'
    LIF_R = 'LIF-R'
    LIF_ASC = 'LIF-ASC'
    LIF_R_ASC = 'LIF-R-ASC'
    LIF_R_ASC_A = 'LIF-R-ASC-A'
    model_id2name = {395310469: LIF, 395310479: LIF_R, 395310475: LIF_ASC, 471355161: LIF_R_ASC, 395310498: LIF_R_ASC_A}
    glif_api = GlifApi()
    
    for cell_result in glif_api.get_neuronal_models(cell_ids):  #[325464516]
        cell_id = cell_result['id']
        for curr_model in cell_result['neuronal_models']:
            if model_id2name[curr_model['neuronal_model_template_id']] != options.model: continue
            model_id = curr_model['id']
            neuron_config = glif_api.get_neuron_configs([model_id])[model_id]
            for stim in options.stimulus.split(','):
                simulate = stimulus[stim]
                output = simulate(cell_id, options.model,neuron_config)
                plt.figure('Cell '+str(cell_id)+' '+options.model+' '+stim)
                plotter.plt_comparison_neurons(np.array(output['I']) * 1.0e12,output['times'], output['voltages'], output['spike_times'], show=False)
    plt.show()
