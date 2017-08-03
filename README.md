# Glif Models Implementation in NEST Simulator

## Build and install modules dynamically
```bash
$ mkdir build
$ cd build
$ cmake --Dwith-nest=nest-config -Dwith-ltdl=ON [-Dwith-mpi=ON] ../GlifModule
$ make
$ make install
```
#### Issues
* Pull and compile nest from the latest on their github repository. The v2.10.0 isn't working.
* Make sure ltdl-dev libraries are available (before compiling nest). On CentOS run ```sudo yum install libtool-ltdl-devel```, on ubuntu ```libltdl-dev```.
* When compiling nest make sure to use absolute paths. When completed run ```nest-config --libs``` to make sure a full path to the nest libraries are used.

## Instantiate Modules in pynest
```python
import nest
nest.Install('glifmodule')
neuron = nest.Create('glif_lif') # or glif_lif_r, glif_lif_asc, glif_lif_r_asc
```
#### Issues 
* If you get a 'File not found' message when trying to install the module:
  * Try using ```nest.Install('glifmodule.so')``` instead (On CentOS 6 lt_dlopenext() isn't working properly).
  * Check LD_LIBRARY_PATH, if needed set ```export LD_LIBRARY_PATH="/full/path/to/nest/module:$LD_LIBRARY_PATH```

## Running and Testing
### Download Cell-Types-DB models to local machine
In scripts/ folder, run the following command to install 10 specific modules from the paper
```bash
$ python allensdk_helper.py
```
Or to get a specific set of models for a given cell-id
```bash
$ python allensdk_helper.py -c CELL-ID1[,CELL-ID2,...]
```

### Test all downloaded models
```bash
$ python test_glif2nest.py 1> /dev/null
```

### Run and qualitativly compare NEST and AllenSDK implementation
First determine the type in injection schemes are available
```bash
$ python run_model.py --list-stimuli
```
The following will run both NEST and AllenSDK implementation of a model and plot voltage-traces and spike-trains. Model download is not required.
```bash
$ python run_model.py --cells cell-id[,cell_id,...] --model LIF[-R|-ASC|-R-ASC|-R-ASC-A] --stimulus ramp-1[,long-square-1,ramp-2,...]
```

### Run NEST implementation of Glif models with current-based synaptic ports 
First determine the type in injection schemes are available
```bash
$ python run_model_psc.py --list-stimuli
```
The following will run NEST implementation of a 4 neurons network as described below and plot voltage-traces and spike-trains. Model download is not required.
* One neuron is without synaptic port, the other three are with 2 syaptic ports (one port is 2.0ms and one port is 1.0ms);
* The first neuron is connected the first port of the second neuron;
* The first neuron is connected the second port of the third neuron;
* The first neuron is also connected both ports of the fourth neuron;
* The weights between first neuron and other neurons are all 1000.0.
```bash
$ python run_model_psc.py --cells cell-id[,cell_id,...] --model LIF[-R|-ASC|-R-ASC|-R-ASC-A] --stimulus ramp-1[,long-square-1,ramp-2,...]
```

### Run NEST implementation of Glif models with conductance-based synaptic ports 
First determine the type in injection schemes are available
```bash
$ python run_model_cond.py --list-stimuli
```
The following will run NEST implementation of a 4 neurons network as described below and plot voltage-traces and spike-trains. Model download is not required.
* One neuron is without synaptic port, the other three are with 2 syaptic ports (one port is 2.0ms and one port is 1.0ms);
* The first neuron is connected the first port of the second neuron;
* The first neuron is connected the second port of the third neuron;
* The first neuron is also connected both ports of the fourth neuron;
* The weights between first neuron and other neurons are all 30.0.
```bash
$ python run_model_cond.py --cells cell-id[,cell_id,...] --model LIF[-R|-ASC|-R-ASC|-R-ASC-A] --stimulus ramp-1[,long-square-1,ramp-2,...]
```

## Notes
* Has only been tested with python 2.7
