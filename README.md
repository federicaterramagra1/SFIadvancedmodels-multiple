# Fault Injection Tool for the Reliability Assessment of Deep Learning Algorithms

## Overview
**SFIadvancedmodels** is an open-source software designed to test the resilience of deep learning algorithms against the occurrence of random-hardware faults. The intent of the framework is to execute advanced statistical fault injection analyses by extending the available and known fault models in the literature.

## Projects structure

This project is structured as follows:

- `main.py`: The main entry point for our application. It performs FI campaigns and save the OFM and the outputs (golden and faulty)
- `SETTINGS.py`: Configuration file
- `utils.py`: Utility functions and helper modules
- `faultManager/`: Contains the files used to manage the FI campaigns
- `ofmapManager/`: Saves the OFM of the golden network
- `smartlayers/`: Contains the file usefull to save the faulty OFM
- `dlModels/`: Directory where models and weights are stored

# Setup

To get started, first clone the repository from GitHub:

`git clone https://github.com/your-username/SFIadvancedmodels.git`

## Creating a Python Environment
It is recommended to create a virtual environment to manage your dependencies. You can do this using venv:

`python3 -m venv environment_name`

`source environment_name/bin/activate`

## Installing Dependencies
Once your virtual environment is activated, install the required packages listed in requirements.txt:

`pip install -r requirements.txt`

# Usage
To start a fault injection, compile the ```SETTINGS.py``` file to configure your experiments, then run:

``` python3 main.py ```

The output of the SFI is stored in the folder `output`. More in details:

- `output/clean_feature_maps`: Stores the clean feature maps
- `output/clean_ouput`: Stores the clean output
- `outpput/fault_list`: The fault list used for the injections
- `output/faulty_feature_maps` : Stores the faulty feature maps
- `output/faulty_ouput`: Stores the faulty output
- `results/`: Stores the analysis of the outputs

The file are named as follow:

- clean FM: ```batch_[batch_id]_layer_[layer_name].npz```. 
This file contains the clean output feature map of layer `[layer_id]` given the input batch `[batch_id]`.
- clean output: ```clean_output.npy```.
This file contains the clean output for all the input batches.
- faulty FM: ```fault_[fault_id]_batch_[batch_id]_layer_[layer_name].npz```.
This file contains the faulty output feature map of layer `[layer_id]` given the input batch `[batch_id]` when the fault
`[fault_id]` is injected.
- faulty output: ```[fault_model]/batch_[batch_id].npy```.
This file contains the clean output given the input batch `[batch_id]` for all the faults injected.

The files are either np or npz array. The dimensions are the following:

- clean FM: ```BxKxHxW```
- clean output: ```NxBxC```
- faulty FM: ```BxKxHxW```
- clean output: ```FxBxC```

Where `F` is the length of the fault list, `N` is the number of batches, `B` is the batch size, `C` is the number of
classes, `K` is the number of channels of an OFM, `H` is the height of an OFM and `W` is the width.

To load the FM arrays call ```np.load(file_name)['arr_0'])```. To load the output array call ```np.load(file_name, allow_pickle=True)```.

