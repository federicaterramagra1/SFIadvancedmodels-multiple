import torch


# SET THE NETWORKS 

from dlModels.CIFAR10 import resnet_cifar10,mobilenetv2_cifar10


'''
FAULT_MODEL available: 'stuck-at_params', 'byzantine_neuron'


NETWORK available: 'ResNet18', 'ResNet20', 'ResNet32', 'ResNet44', 
                          'DenseNet121', 'DenseNet161','MobileNetV2', 
                          'GoogLeNet', 'Vgg11_bn', 'Vgg13_bn'


DATASET available: 'CIFAR10', 'CIFAR100', 'GTSRB'

'''
# ----------------- FAULT LIST SETTINGS -----------------

# enable the fault list generation
FAULT_LIST_GENERATION = True

# FAULT LIST
SEED = 40
DATASET_NAME = 'CIFAR10'
NETWORK_NAME = 'ResNet20'

# FAULT LIST PARAMETERS
error_margin = 0.01
probability = 0.5
confidence_constant = 2.58
bit = 32

modules_to_fault = (torch.nn.Conv2d, torch.nn.Linear)

FAULT_LIST_PATH = f'output/fault_list/{DATASET_NAME}/{NETWORK_NAME}/'
FAULT_LIST_NAME = f'{NETWORK_NAME}_{SEED}_fault_list.csv'

# ----------------- FAULT INJECTION SETTINGS -----------------

# enable the fault injection
FAULTS_INJECTION = True

#fault to inject in the model from the faul list
FAULTS_TO_INJECT = 10

# disable the usage of CUDA ----- SERVE?
FORBID_CUDA = False

# use the GPU is available
USE_CUDA = True

# force the computation of the feature maps ----- SERVE?

# forbif the logging of the results ----- MI SEMBRA NON FUNZIONI
NO_LOG_RESULTS = False

# test set batch size
BATCH_SIZE = 128

# fault model to use (check the top of the file for the available models)
FAULT_MODEL = 'stuck-at_params'

# dataset to use (check the top of the file for the available datasets)
DATASET = 'CIFAR10'

# network to use (check the top of the file for the available networks)
NETWORK = 'ResNet20'

# threshold under which an error is undetected
THRESHOLD = 0.0

# gaussian filter to the ofm to decrease fault impact  ----- SERVE?
GAUSSIAN_FILTER = False

FORCE_RELOAD = False

# ANALYSIS
FI_ANALYSIS = True

# ------- SAVE SETTINGS -------

# SAVE CLEAN OFM
SAVE_CLEAN_OFM = False

# SAVE FAULTY OFM
SAVE_FAULTY_OFM = False

# SAVE FAULTY OUTPUT
SAVE_FAULTY_OUTPUT = True

# OFM TO SAVE
if SAVE_FAULTY_OFM:   
    INPUT_FMAPS_TO_SAVE = 'layer1.0.conv1'
else:
    INPUT_FMAPS_TO_SAVE = None

# ------- PATHS -------

# CLEAN FOLDER PATHS
CLEAN_FM_FOLDER = f'output/clean_feature_maps/{DATASET}/{NETWORK}/batch_{BATCH_SIZE}'
CLEAN_OUTPUT_FOLDER = f'output/clean_output/{DATASET}/{NETWORK}/batch_{BATCH_SIZE}'

# FAULTY FOLDER PATHS
FAULTY_FM_FOLDER = f'output/faulty_feature_maps/{DATASET}/{NETWORK}/batch_{BATCH_SIZE}/{FAULT_MODEL}'
FAULTY_OUTPUT_FOLDER = f'output/faulty_output/{DATASET}/{NETWORK}/batch_{BATCH_SIZE}'

# MODULES TO SAVE OFM AND IFM
MODULE_CLASSES = (torch.nn.Conv2d)
MODULE_CLASSES_FAULT_LIST = (torch.nn.Conv2d)

# DATASET PATHS
DATASET_PATH = f'Datasets/'

# MODELS PATHS
MODELS_PATH = f'dlModels/'

# PRETRAINED MODEL PATHS
MODEL_TH_PATH = f'dlModels/{DATASET}/pretrained/{NETWORK}_{DATASET}.th'
MODEL_PT_PATH = f'dlModels/{DATASET}/pretrained/{NETWORK}_{DATASET}.pt'
MODEL_PTH_PATH = f'dlModels/{DATASET}/pretrained/{NETWORK}_{DATASET}.pth'


FI_ANALYSIS_PATH = f'results/{DATASET}/{NETWORK}/batch_{BATCH_SIZE}/'













