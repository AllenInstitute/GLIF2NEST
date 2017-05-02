import os
import glob
import re
import requests
import json
import pandas as pd
from optparse import OptionParser

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from allensdk.api.queries.cell_types_api import CellTypesApi
from allensdk.api.queries.glif_api import GlifApi
import allensdk.core.json_utilities as json_utilities

LIF = 'LIF'
LIF_R = 'LIF-R'
LIF_ASC = 'LIF-ASC'
LIF_R_ASC = 'LIF-R-ASC'
LIF_R_ASC_A = 'LIF-R-ASC-A'
model_id2name = {395310469: LIF, 395310479: LIF_R, 395310475: LIF_ASC, 471355161: LIF_R_ASC, 395310498: LIF_R_ASC_A}


def download_glif_models(cell_ids, base_dir, incl_ephys=True, force_overwrite=False):
    """Goes through the list of cell_ids and downloads cell config and ephys data in base_dir/cell_<ID>. Then looks up
    all possible models and downloads model files int base_dir/cell_<ID>/<MODEL_TYPE>_<MODEL_ID>/
    """
    # Determine the best url for connecting to cell-types db
    try:
        # see if we can connect to interal cell-types db
        request = requests.get('http://icelltypes/')
        if request.status_code == 200:
            base_uri = 'http://icelltypes/' 
        else:
            base_uri = None 
    except Exception:
        base_uri = None # use the default url

    base_dir = base_dir if base_dir.endswith('/') else base_dir + '/'

    valid_cells = []
    ct_api = CellTypesApi(base_uri)
    for cell in ct_api.list_cells():
        if cell['id'] in cell_ids:
            # create directory for cell
            cell_home = '{}cell_{}/'.format(base_dir, cell['id'])
            if not os.path.exists(cell_home):
                os.makedirs(cell_home)

            # save metadata
            cell_metadata_file = cell_home + 'cell_metadata.json'
            if force_overwrite or not os.path.exists(cell_metadata_file):
                print('Saving metadata for cell {} in {}'.format(cell['id'], cell_metadata_file))
                json_utilities.write(cell_metadata_file, cell)
            else:
                print('File {} already exists. Skipping'.format(cell_metadata_file))

            # save ephys data
            if incl_ephys:
                cell_ephys_file = cell_home + 'ephys_data.nwb'
                if force_overwrite or not os.path.exists(cell_ephys_file):
                    print('Saving ephys data for cell {} in {}'.format(cell['id'], cell_ephys_file))
                    ct_api.save_ephys_data(cell['id'], cell_ephys_file)
                else:
                    print('File {} already exists. Skipping'.format(cell_ephys_file))

            # save sweeps file
            sweeps_file = cell_home + 'ephys_sweeps.json'
            if force_overwrite or not os.path.exists(sweeps_file):
                print('- Saving sweeps file to {}'.format(sweeps_file))
                ephys_sweeps = ct_api.get_ephys_sweeps(cell['id'])
                json_utilities.write(sweeps_file, ephys_sweeps)
            else:
                print('- File {} already exits. Skipping.'.format(sweeps_file))

            # keep track of valid ids
            valid_cells.append(cell['id'])
            cell_ids.remove(cell['id'])

    for cid in cell_ids:
        print('Warning: cell #{} was not found in cell-types database'.format(cid))

    # Iterate through each all available models and find ones correspoding to cell list
    glif_models = {} # map model-id to their directory
    glif_api = GlifApi(base_uri=base_uri)
    for model in glif_api.list_neuronal_models():
        if model['specimen_id'] in valid_cells:
            # save model files <BASE_DIR>/cell_<CELL_ID>/<MODEL_TYPE>_<MODEL_ID>/ 
            cell_id = model['specimen_id']
            model_id = model['id']
            model_type = model['neuronal_model_template_id']#['id'] # type of model, GLIF-LIF, GLIF-ASC, etc
            type_name = model_id2name.get(model_type, None)
            if type_name is None:
                print('Warning: Unknown model type {} ({}) for cell/model {}/{}'.format(model_type, 
                                                                                        model['neuronal_model_template']['name'],
                                                                                        cell_id, model_id))
                type_name = model_type
            model_home_dir = '{}cell_{}/{}_{}/'.format(base_dir, cell_id, type_name, model_id) 
            glif_models[model_id] = model_home_dir
            

    # go through all the found models, download necessary files
    n_models = len(glif_models)
    for i, (gid, home_dir) in enumerate(glif_models.iteritems()):
        print('Processing model {}  ({} of {})'.format(gid, (i+1), n_models))
        model_metadata = glif_api.get_neuronal_model(gid)

        if not os.path.exists(home_dir):
            os.makedirs(home_dir)

        # save model metadata
        metadata_file = home_dir + 'metadata.json'
        if force_overwrite or not os.path.exists(metadata_file):
            print('- Saving metadata file to {}'.format(metadata_file))
            #print type(metadata_file)
            with open(metadata_file, 'wb') as fp:
                json.dump(model_metadata, fp, indent=2)
        else:
            print('- File {} already exits. Skipping.'.format(metadata_file))

        # get neuron configuration file
        config_file = home_dir + 'config.json'
        if force_overwrite or not os.path.exists(config_file):
            print('- Saving configuration file to {}'.format(config_file))
            neuron_config = glif_api.get_neuron_config()
            json_utilities.write(config_file, neuron_config)
        else:
            print('- File {} already exits. Skipping.'.format(config_file))



def get_models_dir(base_dir='../models', cell_id=None, model_type=None, model_id=None):
    """Returns a list of model directory locations in format (CELL_ID, MODEL_TYPE, MODEL_ID, PATH_TO_MODEL_FILES)
    using a combination of cell_id, model_type and/or model_id.

    Letting cell_id, model_type, or model_id be None is equivilent to set it to *
    """
    base_dir = base_dir[:-1] if base_dir.endswith('/') else base_dir
    if not os.path.isdir(base_dir):
        print('Could not find base-directory {}.'.format(base_dir))
        return []

    # Find all directories matching patterns
    cell_str = 'cell_*' if cell_id is None else 'cell_' + str(cell_id)
    type_str = '*' if model_type is None else model_type
    id_str = '*' if model_id is None else model_id
    path_pattern = '{}/{}/{}_{}'.format(base_dir, cell_str, type_str, id_str)
    paths = [n for n in glob.glob(path_pattern) if os.path.isdir(n)]


    # parse path name to find cell id, model id and type, and create tuple
    ret = []
    re_pattern = base_dir + '/cell_(.*)/(.*)_(.*)'
    for p in paths:
        restr = re.search(re_pattern, p)
        cell_id = restr.group(1)
        model_type = restr.group(2)
        model_id = restr.group(3)
        cell_dir = '{}/cell_{}'.format(base_dir, cell_id)
        model_dir = '{}/cell_{}/{}_{}'.format(base_dir, cell_id, model_type, model_id)
        ret.append({'speciment': cell_id,
                    'model-type': model_type,
                    'model-id': model_id,
                    'ephys-file': cell_dir + '/ephys_data.nwb',
                    'sweeps-file': cell_dir + '/ephys_sweeps.json',
                    'cell-metadata-file': cell_dir + '/cell_metadata.json',
                    'model-config-file': model_dir + '/config.json',
                    'model-metadata-file': model_dir + '/metadata.json'})
    return ret


def get_model_params(cell_id=None, model_type=None, model_id=None, base_dir='../models', default_val='...'):
    def params2str(d):
        """A helper function for turning a dictionary of parameters into a string"""
        assert (isinstance(d, dict))
        param_str = ''
        if len(d) == 0:
            return param_str

        for k, v in d.iteritems():
            param_str += '{}={}, '.format(k, str(v))
        param_str = param_str[:-2]
        return param_str

    # Find all matching models
    models = get_models_dir(cell_id=cell_id, model_type=model_type, model_id=model_id, base_dir=base_dir)
    if len(models) == 0:
        print('No matching models found.')
        return None

    # A key dictionary of model properties, each key holds a list
    config_params = {'cell-id': [c['speciment'] for c in models],
                     'model-type': [c['model-type'] for c in models],
                     'model-id': [c['model-id'] for c in models]}

    # Go through all the model/config.json files and get parameters
    config_files = [m['model-config-file'] for m in models]
    n_models = len(config_files)
    for i, config in enumerate(config_files):
        with open(config) as fp:
            cdata = json.load(fp)

        # Check if the keys exists in dictionary, if not add.
        for k in cdata.keys():
            if k not in config_params.keys():
                config_params[k] = [default_val for _ in xrange(n_models)]

        # Go through the keys and turn into string values
        for k, v in cdata.iteritems():
            v_str = default_val
            if k == 'coeffs':
                v_str = '[' + params2str(v) + ']'

            elif isinstance(v, dict) and 'name' in v.keys() and 'params' in v.keys():
                if v['name'] != 'none':
                    v_str = '{}({})'.format(v['name'], params2str(v['params']))
            else:
                v_str = str(v)

            config_params[k][i] = v_str

    return pd.DataFrame.from_dict(config_params)


if __name__ == '__main__':
    parser = OptionParser()
    options, args = parser.parse_args()

    if not args:
        # list provide by Corinne, http://biorxiv.org/content/early/2017/01/31/104703
        cell_ids = [314822529, # Rorb
                    510106222, # Ctgf
                    509604672, # Vip
                    527464013, # Ndnf
                    490376252, # Cux2
                    519220630, # Chat
                    485938494, # Ntsr1
                    490205998, # Scnn1a-Tg2
                    323834998, # Scnn1a-Tg3
                    469704261, # Nr5a1
                    477975366, # Htr3a
                    313862134, # Sst
                    490944352] # Rbp4
    else:
        cell_ids = args

    download_glif_models(cell_ids, '../models/')
