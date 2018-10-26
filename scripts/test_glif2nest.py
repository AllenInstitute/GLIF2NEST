"""
Tests the NEST glif models. Before running you need to first download the model files using allensdk_helper.py,
and pipe the NEST output to /dev/null. To run in command line:
 $ python allensdk_helper.py
 $ python test_glif2nest.py 1> /dev/null

The tests first run the model using AllenSDK, then the NEST implementation, and compares spike-trains. (Comparision
using voltage traces are not yet implemented and may be difficult due to AllenSDK not having values during the
refractory period).

The stimuli and functions for the models are stored in run_model.py, and tests are automatically
generated. To add a test create a new entry in run_model.py's stimulus_table.
"""
import unittest

import allensdk_helper as asdk
import run_model as models
import json

# Change this to the location where the cell files are located
BASE_DIR = '../models'


class Test_LIF(unittest.TestCase):
    """LIF tests"""
    longMessage = True
    model_id = 'LIF'


class Test_LIF_ASC(unittest.TestCase):
    """LIF-ASC tests"""
    longMessage = True
    model_id = 'LIF-ASC'


class Test_LIF_R(unittest.TestCase):
    """LIF tests"""
    longMessage = True
    model_id = 'LIF-R'


class Test_LIF_ASC_R(unittest.TestCase):
    """LIF-ASC tests"""
    longMessage = True
    model_id = 'LIF-ASC-R'
    

class Test_LIF_ASC_R_A(unittest.TestCase):
    """LIF-ASC tests"""
    longMessage = True
    model_id = 'LIF-ASC-R-A'


def make_test_function(cell_id, stim):
    """Dynamically generates a test for the given cell and stimulus

    Parameters
    ----------
    cell_id : An ID of cell that contains config files for the model type
    stim : name of stimulus function + parameters in stimulus-table
    """
    def test(self):
        # Grab stimulus function + params and run for both AllenSDK and NEST models
        sim_fn = models.stimulus[stim]
        m_data_one = asdk.get_models_dir(base_dir=BASE_DIR, cell_id  = cell_id, model_type = self.model_id)
        with open(m_data_one[0]['model-config-file'],'r') as f_config:
            config = json.load(f_config)
        output = sim_fn(cell_id, self.model_id, config)
        nest_vals = output['nest']
        allen_vals = output['allen']

        # Compare the number of spikes
        assert(len(nest_vals['spike_times']) == len(allen_vals['spike_times']))

        # TODO: Implement a better comparision of spike trains
    return test

if __name__ == '__main__':
    # Find all the cell models in the base directory
    m_data = asdk.get_models_dir(base_dir=BASE_DIR)
    cells = [c['speciment'] for c in m_data]

    # Generate a test for every combination of cell, model, and stimulus
    for cell_id in cells:
        for name in models.stimulus.keys():
            test_func = make_test_function(cell_id, name)
            setattr(Test_LIF, 'test_{}_{}'.format(cell_id, name), test_func)

        for name in models.stimulus.keys():
            test_func = make_test_function(cell_id, name)
            setattr(Test_LIF_ASC, 'test_{}_{}'.format(cell_id, name), test_func)
            
        for name in models.stimulus.keys():
            test_func = make_test_function(cell_id, name)
            setattr(Test_LIF_R, 'test_{}_{}'.format(cell_id, name), test_func)

        for name in models.stimulus.keys():
            test_func = make_test_function(cell_id, name)
            setattr(Test_LIF_ASC_R, 'test_{}_{}'.format(cell_id, name), test_func)
            
        for name in models.stimulus.keys():
            test_func = make_test_function(cell_id, name)
            setattr(Test_LIF_ASC_R_A, 'test_{}_{}'.format(cell_id, name), test_func)

    unittest.main(verbosity=2)
