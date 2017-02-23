import os
import numpy as np
import matplotlib.pyplot as plt

from allensdk.model.glif.glif_neuron import GlifNeuron
import allensdk.core.json_utilities as json_utilities
from allensdk.core.nwb_data_set import NwbDataSet


import allensdk_helper as asdk
from plot_helper import plot_vt



def run_sweep(cell_id, model_type, sweep_number, base_dir='../models'):
    m_data = asdk.get_models_dir(base_dir=base_dir, cell_id=cell_id, model_type=model_type)

    if len(m_data) == 0:
        print("Couldn't find a {} model for cell {} in {}.".format(model_type, cell_id, base_dir))
        return

    if len(m_data) > 1:
        print('Found more than one {} model for cell {}.'.format(model_type, cell_id))
        return

    model_files = m_data[0]
    model_config_file = model_files['model-config-file']
    ephys_file = model_files['ephys-file']
    sweeps_file = model_files['sweeps-file']

    neuron_config = json_utilities.read(model_config_file)
    neuron = GlifNeuron.from_dict(neuron_config)


    for c_name, m_type, m_id, m_path in m_data:
        print('Model: {}'.format(m_id))
        neuron_config = json_utilities.read(os.path.join(m_path, 'config.json'))
        neuron = GlifNeuron.from_dict(neuron_config)

        ephys_sweeps = json_utilities.read(os.path.join(m_path, 'ephys_sweeps.json'))
        #print ephys_sweeps[0].keys()
        selected_sweep = None
        for s in ephys_sweeps:
            if s['sweep_number'] == 37:
                selected_sweep = s
                break
        ds = NwbDataSet(os.path.join(m_path, 'stimulus.nwb'))
        data = ds.get_sweep(37)



        I = data['stimulus']
        neuron.dt = 1.0/data['sampling_rate']

        ir = data['index_range']

        for k, v in data.iteritems():
            print('{} = {}'.format(k, v))

        print('----------')



        #ephys_file_name = os.path.join(m_path, 'stimulus.nwb')
        #simulate_neuron(neuron, [37], ephys_file_name, ephys_file_name, 0.05)
        #exit()

        output = neuron.run(I)
        voltage = output['voltage']
        #threshold = output['threshold']
        spike_times = output['interpolated_spike_times']

        for k, v in selected_sweep.iteritems():
            print('{} = {}'.format(k, v))
        #print selected_sweep['stimulus_start_time']
        ts = np.arange(len(I))*neuron.dt
        #ts = np.linspace(0.0, 1.2, len(I))
        print len(I)
        print len(spike_times)
        begin_indx = int(spike_times[1]/neuron.dt)
        end_index = int(spike_times[3]/neuron.dt)

        print(begin_indx, end_index)

        plt.plot(voltage[begin_indx:end_index])
        plt.show()

        #print len(ts)
        plot_vt(ts[ir[0]:ir[1]],
                voltage[ir[0]:ir[1]],
                I[ir[0]:ir[1]])


def reset(voltage_t0, threshold_t0, AScurrents_t0):
    #AScurrents_t1 = self.AScurrent_reset_method(self, AScurrents_t0)
    AScurrents_t1 = np.zeros(len(AScurrents_t0))
    # r = [1.0, 1.0]   
    # new_currents=neuron.asc_amp_array * neuron.coeffs['asc_amp_array'] #neuron.asc_amp_array are amplitudes initiating after the spike is cut
    # left_over_currents=AScurrents_t0 * r * np.exp(-(neuron.k * neuron.dt * neuron.spike_cut_length)) #advancing cut currents though the spike    

    return new_currents+left_over_currents


    #voltage_t1 = self.voltage_reset_method(self, voltage_t0)
    voltage_t1 = 0.0

    #threshold_t1 = self.threshold_reset_method(self, threshold_t0, voltage_t1)
    threshold_t1 = threshold_t0

    bad_reset_flag=False
    if voltage_t1 > threshold_t1:
        bad_reset_flag=True
        #TODO put this back in eventually but would rather debug right now
#            raise GlifBadResetException("Voltage reset above threshold: voltage_t1 (%f) threshold_t1 (%f), voltage_t0 (%f) threshold_t0 (%f) AScurrents_t0 (%s)" % ( voltage_t1, threshold_t1, voltage_t0, threshold_t0, repr(AScurrents_t0)), voltage_t1 - threshold_t1)

    return voltage_t1, threshold_t1, AScurrents_t1, bad_reset_flag


def interpolate_spike_value(dt, interpolated_spike_time_offset, v0, v1):
    """ Take a value at two adjacent time steps and linearly interpolate what the value would be
    at an offset between the two time steps. """
    return v0 + (v1 - v0) * interpolated_spike_time_offset / dt


def line_crossing_x(dx, a0, a1, b0, b1):
    return dx * (b0 - a0) / ((a1 - a0) - (b1 - b0))


def interpolate_spike_time(dt, time_step, threshold_t0, threshold_t1, voltage_t0, voltage_t1):
    #print '({} - {})/(({} - {}) - ({} - {}))'.format(threshold_t0, voltage_t0, voltage_t1, voltage_t0, threshold_t1, threshold_t0)
    #print line_crossing_x(dt, voltage_t0, voltage_t1, threshold_t0, threshold_t1)
    #exit()

    return time_step*dt + line_crossing_x(dt, voltage_t0, voltage_t1, threshold_t0, threshold_t1)


def run_glif_neuron(cell_id, sweep_number, model_type, base_dir='../models'):
    m_data = asdk.get_models_dir(base_dir=base_dir, cell_id=cell_id, model_type=model_type)

    if len(m_data) == 0:
        print("Couldn't find a {} model for cell {} in {}.".format(model_type, cell_id, base_dir))
        return

    if len(m_data) > 1:
        print('Found more than one {} model for cell {}.'.format(model_type, cell_id))
        return

    model_files = m_data[0]
    model_config_file = model_files['model-config-file']
    ephys_file = model_files['ephys-file']
    sweeps_file = model_files['sweeps-file']

    neuron_config = json_utilities.read(model_config_file)
    neuron = GlifNeuron.from_dict(neuron_config)
    datasets = NwbDataSet(ephys_file)

    sweep = datasets.get_sweep(sweep_number)

    #ds = NwbDataSet(os.path.join(m_path, 'stimulus.nwb'))
    #data = ds.get_sweep(37)

    I = sweep['stimulus']
    neuron.dt = 1.0 / sweep['sampling_rate']

    output = neuron.run(I)
    output.update({'stimulus': I})
    output.update({'times': [t*neuron.dt for t in xrange(len(I))]})
    return output


def run_lif(cell_id, sweep_number, model_id=None, base_dir='../models'):
    model_type = asdk.LIF
    models = asdk.get_models_dir(base_dir=base_dir, cell_id=cell_id, model_type=model_type, model_id=model_id)

    if len(models) == 0:
        print("Couldn't find a {} model for cell {} in {}".format(model_type, cell_id, base_dir))

    if len(models) > 1:
        print("Found more than one {} for cell {} in {}, using first one (specify model_id)".format(model_type,
                                                                                                    cell_id, base_dir))
    model = models[0]

    sweeps = json_utilities.read(model['sweeps-file'])
    #print(sweeps[0].keys())
    #exit()

    datasets = NwbDataSet(model['ephys-file'])
    neuron_config = json_utilities.read(model['model-config-file'])

    sweep = datasets.get_sweep(sweep_number)
    stimulus = sweep['stimulus']
    dt = 1.0 / sweep['sampling_rate']
    n_steps = len(stimulus)

    voltage_t0 = neuron_config['init_voltage']
    threshold_t0 = neuron_config['init_threshold']
    AScurrents_t0 = neuron_config['init_AScurrents']

    R_input = neuron_config['R_input']
    coeffs_G = neuron_config['coeffs']['G']
    G = 1.0 / R_input

    El = neuron_config['El']

    coeff_th_inf = neuron_config['coeffs']['th_inf']
    th_inf = neuron_config['th_inf']

    coeff_C = neuron_config['coeffs']['C']
    C = neuron_config['C']




    spike_cut_length = neuron_config['spike_cut_length']

    voltage_out = np.empty(n_steps)
    voltage_out[:] = np.nan
    threshold_out = np.empty(n_steps)
    threshold_out[:] = np.nan
    AScurrents_out = np.empty(shape=(n_steps, len(AScurrents_t0)))
    AScurrents_out[:] = np.nan

    spike_time_steps = []
    grid_spike_times = []
    interpolated_spike_times = []
    interpolated_spike_voltage = []
    interpolated_spike_threshold = []

    threshold = coeff_th_inf * th_inf

    print coeff_th_inf * th_inf
    print G * coeffs_G
    print El
    print C * coeff_C
    print spike_cut_length
    print dt
    print spike_cut_length*dt
    print max(stimulus)
    print len(stimulus)
    print dt*1000
    print sweep['sampling_rate']
    print sweep.keys()
    print len(stimulus)/sweep['sampling_rate']
    print len(stimulus)*dt
    #exit()

    time_step = 0
    while time_step < n_steps:
        inj = stimulus[time_step]

        AScurrents_t1 = np.zeros(len(AScurrents_t0))
        # exp()
        ## return AScurrents_t0*np.exp(-neuron.k*neuron.dt) 

        voltage_t1 = voltage_t0 + (inj + np.sum(AScurrents_t0) - G * coeffs_G * (voltage_t0 - El)) * dt / (C * coeff_C)
        threshold_t1 = coeff_th_inf * th_inf

        if voltage_t1 > threshold_t1:
            #print inj
            print dt
            print inj

            # spike_time_steps are stimulus indices when voltage surpassed threshold
            spike_time_steps.append(time_step)
            grid_spike_times.append(time_step * dt)

            # interpolate exact time when threshold is reached
            spike_time = dt*(time_step + (threshold_t1 - voltage_t0)/(voltage_t1 - voltage_t0))
            interpolated_spike_times.append(spike_time)
            #interpolated_spike_times.append(interpolate_spike_time(dt, time_step, threshold_t0, threshold_t1, voltage_t0, voltage_t1))

            # save threshold of spike (for LIF this is constant)
            interpolated_spike_voltage.append(threshold)
            interpolated_spike_threshold.append(threshold)
            #interpolated_spike_time_offset = interpolated_spike_times[-1] - (time_step - 1) * dt
            #interpolated_spike_voltage.append(interpolate_spike_value(dt, interpolated_spike_time_offset, voltage_t0, voltage_t1))
            #interpolated_spike_threshold.append(interpolate_spike_value(dt, interpolated_spike_time_offset, threshold_t0, threshold_t1))

            # reset voltage, threshold, and afterspike currents
            (voltage_t0, threshold_t0, AScurrents_t0, bad_reset_flag) = reset(voltage_t1, threshold_t1, AScurrents_t1)

            # if we are not integrating during the spike (which includes right now), insert nans then jump ahead
            if spike_cut_length > 0:
                n = spike_cut_length
                voltage_out[time_step:time_step + n] = np.nan
                threshold_out[time_step:time_step + n] = np.nan
                AScurrents_out[time_step:time_step + n, :] = np.nan

                time_step += spike_cut_length
            else:
                # we are integrating during the spike, so store the reset values
                voltage_out[time_step] = voltage_t0
                threshold_out[time_step] = threshold_t0
                AScurrents_out[time_step,:] = AScurrents_t0
                time_step += 1

            if bad_reset_flag:
                voltage_out[time_step:time_step+5] = voltage_t0
                threshold_out[time_step:time_step+5] = threshold_t0
                AScurrents_out[time_step:time_step+5] = AScurrents_t0
                break
        else:
            # there was no spike, store the next voltages
            voltage_out[time_step] = voltage_t1
            threshold_out[time_step] = threshold
            AScurrents_out[time_step, :] = AScurrents_t1

            voltage_t0 = voltage_t1
            threshold_t0 = threshold_t1
            AScurrents_t0 = AScurrents_t1

            time_step += 1

    return {'voltage': voltage_out,
            'threshold': threshold_out,
            'AScurrents': AScurrents_out,
            'stimulus': stimulus,
            'times': [t*dt for t in xrange(n_steps)],
            'grid_spike_times': np.array(grid_spike_times),
            'interpolated_spike_times': np.array(interpolated_spike_times),
            'spike_time_steps': np.array(spike_time_steps),
            'interpolated_spike_voltage': np.array(interpolated_spike_voltage),
            'interpolated_spike_threshold': np.array(interpolated_spike_threshold)}


def get_interval(ts, voltages, stimulus, interval=(0.0, 100.0)):
    assert(interval[0] < interval[1])
    i0 = 0
    i1 = len(ts) - 1
    for i, t in enumerate(ts):
        if t > interval[0]:
            i0 = i
            break

    for i, t in enumerate(ts):
        if t > interval[1]:
            i1 = i
            break

    return ts[i0:i1], voltages[i0:i1], stimulus[i0:i1]


sweep_n = run_lif(323834998, 44)
(ts_n, voltages_n, stimulus_n) = get_interval(sweep_n['times'], sweep_n['voltage'], sweep_n['stimulus'])


sweep_o = run_glif_neuron(323834998, 44, asdk.LIF)
(ts_o, voltages_o, stimulus_o) = get_interval(sweep_o['times'], sweep_o['voltage'], sweep_o['stimulus'])

#plot_vt(ts, voltages, stimulus)
#exit()


assert(len(sweep_o['spike_time_steps']) == len(sweep_n['spike_time_steps']))
assert(sweep_o['spike_time_steps'][0] == sweep_n['spike_time_steps'][0])
assert(sweep_o['spike_time_steps'][1] == sweep_n['spike_time_steps'][1])
assert(sweep_o['spike_time_steps'][-1] == sweep_n['spike_time_steps'][-1])
assert(sweep_o['spike_time_steps'][-2] == sweep_n['spike_time_steps'][-2])

#print sweep['interpolated_spike_voltage'][0]
assert(abs(sweep_o['interpolated_spike_voltage'][0] - sweep_n['interpolated_spike_voltage'][0]) < 1e-4)
assert(abs(sweep_o['interpolated_spike_voltage'][1] - sweep_n['interpolated_spike_voltage'][1]) < 1e-4)
assert(abs(sweep_o['interpolated_spike_voltage'][-1] - sweep_n['interpolated_spike_voltage'][-1]) < 1e-4)
assert(abs(sweep_o['interpolated_spike_voltage'][-2] - sweep_n['interpolated_spike_voltage'][-2]) < 1e-4)


#print sweep['times'][-1]
#(ts, voltages, stimulus) = get_interval(sweep['times'], sweep['voltage'], sweep['stimulus'])

plt.figure(1)
#plot_vt(ts, voltages, stimulus, show=False)
plot_vt(ts_o, voltages_o, stimulus_o, show=False)

plt.figure(2)
plot_vt(ts_n, voltages_n, stimulus_n, show=False)

plt.show()

