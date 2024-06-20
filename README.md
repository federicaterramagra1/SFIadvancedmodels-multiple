# Fault Injection Tool for the Reliability Assessment of Deep Learning Algorithms

## Overview
**SFIadvancedmodels** is an open-source software designed to test the resilience of deep learning algorithms against the occurrence of random-hardware faults. The intent of the framework is to execute advanced statistical fault injection analyses by extending the available and known fault models in the literature.

## Projects structure

This project is structured as follows:
- `requirements.txt`: packages to install in a virtual environment to run the application
- `main.py`: The main entry point for our application. It performs fault list generations, FI campaigns where it saves the OFM and the outputs (golden and faulty) and a final FI analysis
- `SETTINGS.py`: Configuration file to set preferences
- `utils.py`: Utility functions and helper modules
- `faultManager/`: Contains the files used to manage the FI campaigns
- `ofmapManager/`: Saves the OFM of the golden network
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
To generate the fault list, to start a fault injection, or to analyze the data, compile the ```SETTINGS.py``` file to configure your experiments, then run:

``` python3 main.py ```

It is noted that the type of fault injected is permanent and simulates a stuck-at fault in the memory where the model weights are stored

## Outputs
The code is divided into four individually activatable parts that produce different outputs, controlled by boolean variables in the SETTINGS.py file:

- ```FAULT_LIST_GENERATION```: Generates a fault list for the selected network based on the set parameters.
- ```FAULTS_INJECTION```: Loads the fault list and executes the fault injection campaign, saving outputs or golden/corrupted OFMs based on the preferences set.
- ```FI_ANALYSIS```: Analyzes the corrupted outputs against the golden ones and returns the number of masked, non-critical, and critical (SDC-1) fault.
- ```FI_ANALYSIS_SUMMARY```: When injecting a large number of faults or using large datasets, the previous analysis can produce very large and hard-to-handle CSV files. This variable activates a script that summarizes the previously generated data to make it more accessible.

The output of the SFI is stored in the folder `output`. More in details:

- `output/clean_feature_maps`: Stores the clean feature maps
- `output/clean_ouput`: Stores the clean output
- `outpput/fault_list`: The fault list used for the injections
- `output/faulty_feature_maps` : Stores the faulty feature maps
- `output/faulty_ouput`: Stores the faulty output
- `results/`: Stores the analysis of the outputs
- `results_summary/`: Stores the summarized analysis of the outputs


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

### Fault list

The generated fault lists are CSV files with a specific format to which the FI refers in order to inject faults into the neural model. The structure is as follows:


FL example for a VGG-11 model with GTSRB dataset

| Injection |    Layer   |   TensorIndex  | Bit |
|:---------:|:----------:|:--------------:|:---:|
|         0 | features.0 | "(3, 0, 2, 1)" |  15 |
|    ...    |     ...    |       ...      | ... |

- `Injection`: column indicating the injection number.
- `Layer`: the layer in which the fault is injected.
- `TensorIndex`: coordinate of the weight where the fault is injected.
- `Bit`: corrupted bit that is flipped.


### Analysis

The analysis files obtained with `FI_ANALYSIS` option are contained in the `results/` folder and are organized by dataset, model, and batch size: `results/dataset-name/model-name/batch-size/`. 
Inside, there are two files:

- `fault_statistics.txt`: A text file where the total number of masked, non-critical, and critical (SDC-1) inferences are saved.
- `output_analysis.csv`:  A CSV file containing all the information regarding the classification of each fault for every inference.

`output_analysis.csv` is organized as follows:

| fault | batch | image | output |
|:-----:|:-----:|:-----:|:------:|
|     0 |     0 |     0 |      1 |
|     0 |     0 |     1 |      0 |
|     0 |     0 |     2 |      0 |
|     0 |     0 |     3 |      2 |
|  ...  |  ...  |  ...  |   ...  |
| 16663 |     9 |  1024 |      1 |
a
- `fault`: Unique identifier of the injected fault, corresponding to the `Injection` column in the fault list used.
- `batch`: Batch containing the dataset images used for inference.
- `image`: Image in the batch on which the inference was performed.
- `output`: Classification of the injected fault by comparing the golden outputs with the corrupted ones obtained from the image inference. The returned values are `0` for a masked fault, `1` for a non-critical fault, and `2` for a critical fault (SDC-1).

### Summarized analysis

Due to the verbosity of the `output_analysis.csv` file, if many faults are injected or a large number of images are used for inferences, the readability of the CSV decreases significantly. To address this issue, using the `FI_ANALYSIS_SUMMARY` option, you can generate a new CSV file named `model-name_summary.csv` inside the `results_summary/dataset-name/model-name/batch-size/` folder. The CSV is organized as follows:

| Injection | Layer |   TensorIndex   | Bit | n_injections | masked | non_critical | critical |
|:---------:|:-----:|:---------------:|:---:|:------------:|:------:|:------------:|:--------:|
|         0 | conv1 |  "(7, 0, 2, 1)" |  15 |        10000 |  10000 |            0 |        0 |
|         1 | conv1 | "(14, 0, 2, 0)" |   5 |        10000 |  10000 |            0 |        0 |
|         2 | conv1 | "(27, 0, 0, 0)" |  13 |        10000 |    701 |         9298 |        1 |
|         3 | conv1 | "(14, 2, 2, 0)" |  12 |        10000 |   9998 |            2 |        0 |