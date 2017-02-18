import os
import requests

from allensdk.api.queries.cell_types_api import CellTypesApi
from allensdk.api.queries.glif_api import GlifApi
import allensdk.core.json_utilities as json_utilities


model_id2name = {395310469: 'LIF', 395310479: 'LIF-R', 395310475: 'LIF-ASC', 471355161: 'LIF-R-ASC', 395310498: 'LIF-R-ASC-A'}

def download_glif_models(cell_ids, base_dir, incl_ephys=True, force_overwrite=False):

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
            if not os.path.exists(model_home_dir):
                os.makedirs(model_home_dir)
            glif_models[model_id] = model_home_dir
            

    # go through all the found models, download necessary files
    for gid, home_dir in glif_models.iteritems():
        print('Processing model {}.'.format(gid))
        glif_api.get_neuronal_model(gid)

        # get stimulus nwb file
        stim_file = home_dir + 'stimulus.nwb'
        if force_overwrite or not os.path.exists(stim_file):
            print('- Saving stimulus file to {}'.format(stim_file))
            glif_api.cache_stimulus_file(stim_file)
        else:
            print('- File {} already exits. Skipping.'.format(stim_file))
            
        # get neuron configuration file
        config_file = home_dir + 'config.json'
        if force_overwrite or not os.path.exists(config_file):
            print('- Saving configuration file to {}'.format(config_file))
            neuron_config = glif_api.get_neuron_config()
            json_utilities.write(config_file, neuron_config)
        else:
            print('- File {} already exits. Skipping.'.format(config_file))

        # get sweeps file
        sweeps_file = home_dir + 'ephys_sweeps.json'
        if force_overwrite or not os.path.exists(sweeps_file):
            print('- Saving sweeps file to {}'.format(sweeps_file))
            ephys_sweeps = glif_api.get_ephys_sweeps()
            json_utilities.write(sweeps_file, ephys_sweeps)
        else:
            print('- File {} already exits. Skipping.'.format(sweeps_file))



if __name__ == '__main__':
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
    download_glif_models(cell_ids, '../models/')


# Download metadata and ephys data for given cell id
#ct = CellTypesApi()
#cells = ct.list_cells()
#print cells[0]['id']
#print cells[0].keys()


#mtemplates = set()
#glif_api = GlifApi()
#glif_api = GlifApi(base_uri=base_uri)
#model_list = glif_api.list_neuronal_models()
#for model in model_list:
#    mt = model['neuronal_model_template']
#    key = '{} - {}'.format(mt['id'], mt['name'])
#    mtemplates.add(key)
#    #print('{} - {}'.format(mt['id'], mt['name']))
#    """
#    if model['specimen_id'] == 313862134:
#        print model['name']
#        print model['id']
#    """

#print mtemplates

#print model_list[0]['specimen_id']
#print model_list[0]['name']
#print model_list[0]['id']
#print model_list[0]['neuronal_model_template']
#for model ini 
