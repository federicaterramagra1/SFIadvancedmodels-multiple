import os

import numpy as np
import pandas as pd
import shutil
from typing import Union, List, Tuple
from functools import reduce

import SETTINGS
import torch
from torch.nn import Sequential, Module

from torch.utils.data import DataLoader

from faultManager.FaultListGenerator import FaultListGenerator
from faultManager.NeuronFault import NeuronFault
from faultManager.WeightFault import WeightFault

from smartLayers.SmartModulesManager import SmartModulesManager
from torchvision.models import resnet
from torchvision.models.densenet import _DenseBlock, _Transition
from torchvision.models.efficientnet import Conv2dNormActivation
from torchvision import transforms
from torchvision.datasets import GTSRB, CIFAR10, CIFAR100, MNIST, ImageNet
from torchvision.transforms.v2 import ToTensor,Resize,Compose,ColorJitter,RandomRotation,AugMix,GaussianBlur,RandomEqualize,RandomHorizontalFlip,RandomVerticalFlip

import csv
from tqdm import tqdm

from dlModels.CIFAR10.mobilenetv2_cifar10 import MobileNetV2
from dlModels.CIFAR10.resnet_cifar10 import resnet20

import random


class UnknownNetworkException(Exception):
    pass


def clean_inference(network, loader, device, network_name):
       

    clean_output_scores = list()
    clean_output_indices = list()
    clean_labels = list()

    counter = 0
    with torch.no_grad():
        
        pbar = tqdm(loader,
                colour='green',
                desc=f'Clean Run',
                ncols=shutil.get_terminal_size().columns)
    
        dataset_size = 0
        
        for batch_id, batch in enumerate(pbar):
            
            data, label = batch
            dataset_size = dataset_size + len(label)
            data = data.to(device)
            
            network_output = network(data)
            prediction = torch.topk(network_output, k=1)
            scores = network_output.cpu()
            indices = [int(fault) for fault in prediction.indices]
            
            clean_output_scores.append(scores)
            clean_output_indices.append(indices)
            clean_labels.append(label)
            
            counter = counter + 1


        elementwise_comparison = [label != index for labels, indices in zip(clean_labels, clean_output_indices) for label, index in zip(labels, indices)]          
        # Count the number of different elements
        num_different_elements = elementwise_comparison.count(True)
        
        print(f'device: {device}')
        print(f'network: {network_name}')
        print(f"The DNN wrong predicions are: {num_different_elements}")
        accuracy= (1 - num_different_elements/dataset_size)*100
        print(f"The final accuracy is: {accuracy}%")
        
        

def get_network(network_name: str,
                device: torch.device,
                dataset_name: str,
                root: str = '.') -> torch.nn.Module:
    
    # Load the network by using the name of the mode and the dataset
    
    if dataset_name == 'CIFAR10':
        print(f'Loading network {network_name} ...')   
        if 'ResNet20' in network_name:
            network = SETTINGS.resnet_cifar10.resnet20() # FIXATO PER IL LABORATORIO
        elif 'ResNet32' in network_name:
            network = SETTINGS.resnet_cifar10.resnet32()
        elif 'ResNet44' in network_name:
            network = SETTINGS.resnet_cifar10.resnet44()
        elif 'DenseNet121' in network_name:
            network = SETTINGS.densenet_cifar10.densenet121()
        elif 'DensenNet161' in network_name:
            network = SETTINGS.densenet_cifar10.densenet161()
        elif 'GoogLeNet' in network_name:
            network = SETTINGS.googlenet_cifar10.googlenet()
        elif 'Vgg11_bn' in network_name:
            network = SETTINGS.vgg_cifar10.vgg11_bn()
        elif 'Vgg13_bn' in network_name:
            network = SETTINGS.vgg_cifar10.vgg13_bn()
        elif 'MobileNetV2' in network_name:
            network = SETTINGS.mobilenetv2_cifar10.MobileNetV2() # FIXATO PER IL LABORATORIO
            
            network_path = SETTINGS.MODEL_PT_PATH
            state_dict = torch.load(network_path, map_location=device)["net"]
            function = None
            if function is None:
                clean_state_dict = {
                    key.replace("module.", ""): value for key, value in state_dict.items()
                }
            else:
                clean_state_dict = {
                    key.replace("module.", ""): function(value)
                    if not (("bn" in key) and ("weight" in key))
                    else value
                    for key, value in state_dict.items()
                }
            network.load_state_dict(clean_state_dict, strict=False)
        else:
            raise ValueError(f'Invalid network name {network}')

        # Load the weights
        if 'MobileNetV2' not in network_name:
            if 'ResNet' in network_name:
                network_path = SETTINGS.MODEL_TH_PATH
            else:
                network_path = SETTINGS.MODEL_PT_PATH
        
            load_from_dict(network=network,
                            device=device,
                            path=network_path)
        
    elif dataset_name == 'CIFAR100':
        print(f'Loading network {network_name} ...')
        if 'ResNet18' in network_name:
            network = SETTINGS.resnet_cifar100.resnet18()
        elif 'DesneNet121' in network_name:
            network = SETTINGS.densenet_cifar100.densenet121()
        elif 'GoogLeNet' in network_name:
            network = SETTINGS.googlenet_cifar100.googlenet()
        elif 'ResNext50' in network_name:
            network = SETTINGS.resnext_cifar100.resnext50()
        else:
            raise ValueError(f'Invalid network name {network}')
        
        # Load the weights
        network_path = SETTINGS.MODEL_PTH_PATH
        function = None
        
        state_dict = torch.load(network_path, map_location=device)['state_dict'] if '.th' in network_path else torch.load(network_path, map_location=device)
        clean_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()} if function is None else {key.replace('module.', ''): function(value) if not (('bn' in key) and ('weight' in key)) else value for key, value in state_dict.items()}
        network.load_state_dict(clean_state_dict, strict=False)

    elif dataset_name == 'GTSRB':
        print(f'Loading network {network_name} ...')
        if 'ResNet20' in network_name:
            network = SETTINGS.resnet_GTSRB.resnet20()
        elif 'DenseNet121' in network_name:
            network = SETTINGS.densenet_GTSRB.densenet121()
        elif 'Vgg11_bn' in network_name:
            network = SETTINGS.vgg_GTSRB.vgg11_bn()
        else:
            raise ValueError(f'Invalid network name {network}')
        
        network_path = SETTINGS.MODEL_PT_PATH
        
        load_from_dict(network=network,
                        device=device,
                        path=network_path)

    network.to(device)
    network.eval()
    
    return network


def get_loader(network_name: str,
               batch_size: int,
               image_per_class: int = None,
               dataset_name: str = None,
               network: torch.nn.Module = None) -> DataLoader:
    """
    Return the loader corresponding to a given network and with a specific batch size
    :param network_name: The name of the network
    :param batch_size: The batch size
    :param image_per_class: How many images to load for each class
    :param network: Default None. The network used to select the image per class. If not None, select the image_per_class
    that maximize this network accuracy. If not specified, images are selected at random
    :return: The DataLoader
    """
    if 'CIFAR10' == dataset_name:
        print('Loading CIFAR10 dataset')
        train_loader, _, loader = load_CIFAR10_datasets(test_batch_size=batch_size,
                                             test_image_per_class=image_per_class)
        
    elif 'CIFAR100' == dataset_name:
        print('Loading CIFAR100 dataset')
        train_loader, _, loader = Load_CIFAR100_datasets(test_batch_size=batch_size,
                                             test_image_per_class=image_per_class)
        
    elif 'GTSRB' == dataset_name:
        print('Loading GTSRB dataset')
        train_loader, _, loader = Load_GTSRB_datasets(test_batch_size=batch_size,
                                             test_image_per_class=image_per_class)
    else:
        print('no dataset specified')
        exit()


    print(f'Batch size:\t\t{batch_size} \nNumber of batches:\t{len(loader)}')

    return train_loader, loader


def get_delayed_start_module(network: Module,
                             network_name: str) -> Module:
    """
    Get the delayed_start_module of the given network
    :param network: The instance of the network where to look for the fault_delayed_start module
    :param network: The name of the network
    :return: An instance of the delayed_start_module
    """

    # The module to change is dependent on the network. This is the module for which to enable delayed start
    if 'LeNet' in network_name:
        delayed_start_module = network
    elif 'ResNet' in network_name:
        delayed_start_module = network
    elif 'MobileNetV2' in network_name:
        delayed_start_module = network.features
        print('delayed_start_module:', delayed_start_module)
    elif 'DenseNet' in network_name:
        delayed_start_module = network.features
    elif 'EfficientNet' in network_name:
        delayed_start_module = network.features
    else:
        raise UnknownNetworkException

    return delayed_start_module


def get_module_classes(network_name: str) -> Union[List[type], type]:
    """
    Get the module_classes of a given network. The module classes represent the classes that can be replaced by smart
    modules in the network. Notice that the instances of these classes that will be replaced are only the children of
    the delayed_start_module
    :param network: The name of the network
    :return: The type of modules (or of a single module) that will should be replaced by smart modules in the target
    network
    """
    if 'LeNet' in network_name:
        module_classes = Sequential
    elif 'MobileNetV2' in network_name:
        module_classes = Sequential
    elif 'ResNet' in network_name:
        if network_name in ['ResNet18', 'ResNet50']:
            module_classes = Sequential
        else:
            module_classes = resnet.BasicBlock
    elif 'DenseNet' in network_name:
        module_classes = (_DenseBlock, _Transition)
    elif 'EfficientNet' in network_name:
        module_classes = (Conv2dNormActivation, Conv2dNormActivation)
    else:
        raise UnknownNetworkException(f'Unknown network {network_name}')

    return module_classes


def get_fault_list(fault_model: str,
                   fault_list_generator: FaultListGenerator,
                   e: float = .01,
                   t: float = 2.58) -> Tuple[Union[List[NeuronFault], List[WeightFault]], List[Module]]:
    """
    Get the fault list corresponding to the specific fault model, using the fault list generator passed as argument
    :param fault_model: The name of the fault model
    :param fault_list_generator: An instance of the fault generator
    :param e: The desired error margin
    :param t: The t related to the desired confidence level
    :return: A tuple of fault_list, injectable_modules. The latter is a list of all the modules that can be injected in
    case of neuron fault injections
    """
    if fault_model == 'byzantine_neuron':
        fault_list = fault_list_generator.get_neuron_fault_list()
    elif fault_model == 'stuck-at_params':
        fault_list = fault_list_generator.get_weight_fault_list()
    else:
        raise ValueError(f'Invalid fault model {fault_model}')

    injectable_modules = fault_list_generator.injectable_output_modules_list

    return fault_list, injectable_modules


def get_device(forbid_cuda: bool,
               use_cuda: bool) -> torch.device:
    """
    Get the device where to perform the fault injection
    :param forbid_cuda: Forbids the usage of cuda. Overrides use_cuda
    :param use_cuda: Whether to use the cuda device or the cpu
    :return: The device where to perform the fault injection
    """

    # Disable gpu if set
    if forbid_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = 'cpu'
        if use_cuda:
            print('WARNING: cuda forcibly disabled even if set_cuda is set')
    # Otherwise, use the appropriate device
    else:
        if use_cuda:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = ''
                print('ERROR: cuda not available even if use-cuda is set')
                exit(-1)
        else:
            device = 'cpu'

    return torch.device(device)


def formatted_print(fault_list: list,
                    network_name: str,
                    batch_size: int,
                    batch_id: int,
                    faulty_prediction_dict: dict,
                    fault_dropping: bool = False,
                    fault_delayed_start: bool = False) -> None:
    """
    A function that prints to csv the results of the fault injection campaign on a single batch
    :param fault_list: A list of the faults
    :param network_name: The name of the network
    :param batch_size: The size of the batch of the data loader
    :param batch_id: The id of the batch
    :param faulty_prediction_dict: A dictionary where the key is the fault index and the value is a list of all the
    top_1 prediction for all the image of the given the batch
    :param fault_dropping: Whether fault dropping is used or not
    :param fault_delayed_start: Whether fault delayed start is used or not
    """

    fault_list_rows = [[fault_id,
                       fault.layer_name,
                        fault.tensor_index[0],
                        fault.tensor_index[1] if len(fault.tensor_index) > 1 else np.nan,
                        fault.tensor_index[2] if len(fault.tensor_index) > 2 else np.nan,
                        fault.tensor_index[3] if len(fault.tensor_index) > 3 else np.nan,
                        fault.bit,
                        fault.value
                        ]
                       for fault_id, fault in enumerate(fault_list)
                       ]

    fault_list_columns = [
        'Fault_ID',
        'Fault_Layer',
        'Fault_Index_0',
        'Fault_Index_1',
        'Fault_Index_2',
        'Fault_Index_3',
        'Fault_Bit',
        'Fault_Value'
    ]

    prediction_rows = [
        [
            fault_id,
            batch_id,
            prediction_id,
            prediction[0],
            prediction[1],
        ]
        for fault_id in faulty_prediction_dict for prediction_id, prediction in enumerate(faulty_prediction_dict[fault_id])
    ]

    prediction_columns = [
        'Fault_ID',
        'Batch_ID',
        'Image_ID',
        'Top_1',
        'Top_Score',
    ]

    fault_list_df = pd.DataFrame(fault_list_rows, columns=fault_list_columns)
    prediction_df = pd.DataFrame(prediction_rows, columns=prediction_columns)

    complete_df = fault_list_df.merge(prediction_df, on='Fault_ID')

    file_prefix = 'combined_' if fault_dropping and fault_delayed_start \
        else 'delayed_' if fault_delayed_start \
        else 'dropping_' if fault_dropping \
        else ''

    output_folder = f'output/fault_campaign_results/{network_name}/{batch_size}'
    os.makedirs(output_folder, exist_ok=True)
    complete_df.to_csv(f'{output_folder}/{file_prefix}fault_injection_batch_{batch_id}.csv', index=False)


def enable_optimizations(
        network: Module,
        delayed_start_module: Union[Module, None],
        module_classes: Union[List[type], type],
        device: torch.device,
        fm_folder: str,
        fault_list_generator: FaultListGenerator,
        fault_list: Union[List[NeuronFault], List[WeightFault]],
        input_size: torch.Size = torch.Size((1, 3, 32, 32)),
        injectable_modules: List[Module] = None,
        fault_delayed_start: bool = True,
        fault_dropping: bool = True):

    # Replace the convolutional layers
    if fault_dropping or fault_delayed_start:

        smart_layers_manager = SmartModulesManager(network=network,
                                                   delayed_start_module=delayed_start_module,
                                                   device=device,
                                                   input_size=input_size)

        if fault_delayed_start:
            # Replace the forward module of the target module to enable delayed start
            smart_layers_manager.replace_module_forward()

        # Replace the smart layers of the network
        smart_modules_list = smart_layers_manager.replace_smart_modules(module_classes=module_classes,
                                                                        fm_folder=fm_folder,
                                                                        fault_list=fault_list)

        # Update the network. Useful to update the list of injectable layers when injecting in the neurons
        if injectable_modules is not None:
            fault_list_generator.update_network(network)
            injectable_modules = fault_list_generator.injectable_output_modules_list

        network.eval()
    else:
        smart_modules_list = None

    return injectable_modules, smart_modules_list




def get_module_by_name(container_module: Module,
                       module_name: str) -> Module:
    """
    Return the instance of the submodule module_name inside the container_module
    :param container_module: The container module that contains the module_name module
    :param module_name: The name of the module to find
    :return: The instance of the submodule with the specified name
    """

    # To fine the actual layer with nested layers (e.g. inside a convolutional layer inside a Basic Block in a
    # ResNet, first separate the layer names using the '.'
    formatted_names = module_name.split(sep='.')

    # Access the nested layer iteratively using itertools.reduce and getattr
    module = reduce(getattr, formatted_names, container_module)

    return module


def load_ImageNet_validation_set(batch_size,
                                 image_per_class=None,
                                 network=None,
                                 imagenet_folder='~/Datasets/ImageNet'):
    """

    :param batch_size:
    :param image_per_class:
    :param network: Default None. The network used to select the image per class. If not None, select the image_per_class
    that maximize this network accuracy. If not specified, images are selected at random
    :param imagenet_folder:
    :return:
    """

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_validation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    validation_dataset_folder = 'tmp'
    validation_dataset_path = f'{validation_dataset_folder}/imagenet_{image_per_class}.pt'

    try:
        if image_per_class is None:
            raise FileNotFoundError

        validation_dataset = torch.load(validation_dataset_path)
        print('Resized Imagenet loaded from disk')

    except FileNotFoundError:
        validation_dataset = ImageNet(root=imagenet_folder,
                                      split='val',
                                      transform=transform_validation)

        if image_per_class is not None:
            selected_validation_list = []
            image_class_counter = [0] * 1000

            # First select only correctly classified images
            for validation_image in tqdm(validation_dataset, desc='Resizing Imagenet Dataset', colour='Yellow'):
                if image_class_counter[validation_image[1]] < image_per_class:
                    prediction = network(validation_image[0].cuda().unsqueeze(dim=0)).argmax() if network is not None else validation_image[1]
                    if prediction == validation_image[1]:
                        selected_validation_list.append(validation_image)
                        image_class_counter[validation_image[1]] += 1

            # Then select images to fill up
            for validation_image in tqdm(validation_dataset, desc='Resizing Imagenet Dataset', colour='Yellow'):
                if image_class_counter[validation_image[1]] < image_per_class:
                    selected_validation_list.append(validation_image)
                    image_class_counter[validation_image[1]] += 1
            validation_dataset = selected_validation_list

        os.makedirs(validation_dataset_folder, exist_ok=True)
        torch.save(validation_dataset, validation_dataset_path)

    # DataLoader is used to load the dataset
    # for training
    val_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                             batch_size=batch_size,
                                             shuffle=False)
    print('Dataset loaded')

    return val_loader


def load_MNIST_datasets(train_batch_size=32, test_batch_size=1):

    train_loader = torch.utils.data.DataLoader(
        MNIST(SETTINGS.DATASET_PATH, train=True, download=True,
              transform=transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Resize((32, 32)),
                  transforms.Normalize(
                      (0.1307,), (0.3081,))
              ])),
        batch_size=train_batch_size, shuffle=False)

    test_loader = torch.utils.data.DataLoader(
        MNIST(SETTINGS.DATASET_PATH, train=False, download=True,
              transform=transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Resize((32, 32)),
                  transforms.Normalize(
                      (0.1307,), (0.3081,))
              ])),
        batch_size=test_batch_size, shuffle=True)

    print('Dataset loaded')

    return train_loader, test_loader



def Load_GTSRB_datasets(train_batch_size=32, train_split=0.8, test_batch_size=1, test_image_per_class=None):
    
    train_transforms = Compose([
    ColorJitter(brightness=1.0, contrast=0.5, saturation=1, hue=0.1),
    RandomEqualize(0.4),
    AugMix(),
    RandomHorizontalFlip(0.3),
    RandomVerticalFlip(0.3),
    GaussianBlur((3,3)),
    RandomRotation(30),
    
    Resize([50,50]),
    ToTensor(),
    transforms.Normalize((0.3403, 0.3121, 0.3214),
                            (0.2724, 0.2608, 0.2669))
    
    ])

    validation_transforms = Compose([
        Resize([50, 50]),
        ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)),
    ])

    train_dataset = GTSRB(root=SETTINGS.DATASET_PATH,
                            split='train',
                            download=True,
                            transform=train_transforms)
    test_dataset = GTSRB(root=SETTINGS.DATASET_PATH,
                            split='test',
                            download=True,
                            transform=validation_transforms)



    # Split the training set into training and validation
    train_split_length = int(len(train_dataset) * 0.8)
    val_split_length = len(train_dataset) - train_split_length
    train_subset, val_subset = torch.utils.data.random_split(train_dataset,
                                                                lengths=[train_split_length, val_split_length],
                                                                generator=torch.Generator().manual_seed(1234))
    # DataLoader is used to load the dataset
    # for training
    train_loader = DataLoader(dataset=train_subset,
                                                batch_size=train_batch_size,
                                                shuffle=True)
    val_loader = DataLoader(dataset=val_subset,
                                                batch_size=train_batch_size,
                                                shuffle=True)  

    test_loader = DataLoader(dataset=test_dataset,
                                                batch_size=test_batch_size,
                                                shuffle=False)

    print('GTSRB Dataset loaded')
        
    return train_loader, val_loader, test_loader

def Load_CIFAR100_datasets(train_batch_size=32, train_split=0.8, test_batch_size=1, test_image_per_class=None):
    
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                         (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])

    train_dataset = CIFAR100(SETTINGS.DATASET_PATH, train=True, transform=transform, download=True)
    test_dataset = CIFAR100(SETTINGS.DATASET_PATH, train=False, transform=transform, download=True)

    train_split = 0.8
    train_split_length = int(len(train_dataset) * train_split)
    val_split_length = len(train_dataset) - train_split_length
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, lengths=[train_split_length, val_split_length], generator=torch.Generator().manual_seed(1234))

    train_loader = DataLoader(dataset=train_subset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_subset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)
    
    print('CIFAR100 Dataset loaded')

    return train_loader, val_loader, test_loader

def load_CIFAR10_datasets(train_batch_size=32, train_split=0.8, test_batch_size=1, test_image_per_class=None):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),                                       # Crop the image to 32x32
        transforms.RandomHorizontalFlip(),                                          # Data Augmentation
        transforms.ToTensor(),                                                      # Transform from image to pytorch tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),   # Normalize the data (stability for training)
    ])
    transform_test = transforms.Compose([
        transforms.CenterCrop(32),                                                  # Crop the image to 32x32
        transforms.ToTensor(),                                                      # Transform from image to pytorch tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),   # Normalize the data (stability for training)
    ])

    train_dataset = CIFAR10(SETTINGS.DATASET_PATH,
                            train=True,
                            transform=transform_train,
                            download=True)
    test_dataset = CIFAR10(SETTINGS.DATASET_PATH,
                           train=False,
                           transform=transform_test,
                           download=True)

    if test_image_per_class is not None:
        selected_test_list = []
        image_class_counter = [0] * 10
        for test_image in test_dataset:
            if image_class_counter[test_image[1]] < test_image_per_class:
                selected_test_list.append(test_image)
                image_class_counter[test_image[1]] += 1
        test_dataset = selected_test_list

    # Split the training set into training and validation
    train_split_length = int(len(train_dataset) * train_split)
    val_split_length = len(train_dataset) - train_split_length
    train_subset, val_subset = torch.utils.data.random_split(train_dataset,
                                                             lengths=[train_split_length, val_split_length],
                                                             generator=torch.Generator().manual_seed(1234))
    # DataLoader is used to load the dataset
    # for training
    train_loader = torch.utils.data.DataLoader(dataset=train_subset,
                                               batch_size=train_batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_subset,
                                             batch_size=train_batch_size,
                                             shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=test_batch_size,
                                              shuffle=False)

    print('CIFAR10 Dataset loaded')

    return train_loader, val_loader, test_loader


def load_from_dict(network, device, path, function=None):
    if '.th' in path:
        state_dict = torch.load(path, map_location=device)['state_dict']
        print('state_dict loaded')
    else:
        state_dict = torch.load(path, map_location=device)
        print('state_dict loaded')

    if function is None:
        clean_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    else:
        clean_state_dict = {key.replace('module.', ''): function(value) if not (('bn' in key) and ('weight' in key)) else value for key, value in state_dict.items()}

    network.load_state_dict(clean_state_dict, strict=False)
    print('state_dict loaded into network')
    
    
def output_definition(network_name, batch_size):
    
    masked = 0
    critical = 0
    not_critical = 0
    output_results_list = []

    # Load clean tensor
    clean_output_path =SETTINGS.CLEAN_OUTPUT_FOLDER + '/clean_output.npy'

    loaded_clean_output = np.load(clean_output_path, allow_pickle=True)

    # print(loaded_clean_output.shape)

    # load faulty tensor
    def count_batch(folder, path):
        files = os.listdir(folder)
        files = [f for f in files if os.path.isfile(os.path.join(folder, f))]
        loaded_file = np.load(path)
        n_outputs = loaded_file.shape[2]
        n_faults = loaded_file.shape[0]
        return len(files), n_outputs, n_faults

    # To define these paths check FaultInjectionManager.py to see the faulty output folder path
    batch_folder = SETTINGS.FAULTY_OUTPUT_FOLDER + f'/{SETTINGS.FAULT_MODEL}'
    batch_path = f'{batch_folder}' + '/batch_0.npy'
    number_of_batch, n_outputs, n_faults = count_batch(batch_folder,  batch_path)

    print(f'number of batch: {number_of_batch}')

    # Define the shape of the tensor
    dim1 = n_faults 
    dim2 = number_of_batch 
    dim3 = int(batch_size) 
    dim4 = n_outputs 


    result_tensor = np.zeros((dim1, dim2, dim3, dim4))
    # print(result_tensor.shape)

    batch_data_list = []


    for i in tqdm(range(number_of_batch)):
        
        file_name = SETTINGS.FAULTY_OUTPUT_FOLDER + f'/{SETTINGS.FAULT_MODEL}' + f'/batch_{i}.npy'
        loaded_faulty_output = np.load(file_name)
        batch_data_list.append(loaded_faulty_output)


    # Find the maximum number of images across all batches
    max_images = max(data.shape[1] for data in batch_data_list)

    # Update dim3 with the maximum number of images
    dim3 = max_images

    # Initialize the result tensor with the correct dimensions
    faulty_tensor_data = np.zeros((n_faults, number_of_batch, dim3, n_outputs))

    # Populate the result tensor with the loaded data
    for i, data in enumerate(batch_data_list):
        result_tensor[:, i, :data.shape[1], :] = data

    print('faulty outputs loaded')
    # print(result_tensor.shape)   
    
 
    faulty_tensor_data = result_tensor
    
    os.makedirs(SETTINGS.FI_ANALYSIS_PATH, exist_ok=True)
    
    # open the .csv
    with open(f'{SETTINGS.FI_ANALYSIS_PATH}/output_analysis.csv', mode='w') as file_csv:

        csv_writer = csv.writer(file_csv)
        csv_writer.writerow(['fault', 'batch', 'image', 'output'])

        print(f'faults: {n_faults}, batches: {number_of_batch}')

        #inside faults
        for z in tqdm(range(dim1), desc="output definition progress"):
            #inside batches
            for i in range(dim2):
                # inside images
                for j in range(min(dim3, loaded_clean_output[i].shape[0])):
                    clean_output_argmax = np.argmax(loaded_clean_output[i][j, :])
                    faulty_output_argmax = np.argmax(faulty_tensor_data[z, i, j, :])                
                    
                    # comparing and save in the .csv the results
                    if np.array_equal(loaded_clean_output[i][j, :], faulty_tensor_data[z, i, j, :]):
                        masked += 1
                        output_results_list.append('0')
                        csv_writer.writerow([z, i, j, '0'])

                    elif clean_output_argmax == faulty_output_argmax:
                        not_critical += 1
                        output_results_list.append('1')
                        csv_writer.writerow([z, i, j, '1'])

                    else:
                        critical += 1
                        output_results_list.append('2')
                        csv_writer.writerow([z, i, j, '2'])
                                                            
        # print the results
        print(f'total outputs: {masked + not_critical + critical}')
        print('masked:', masked)
        print(f'% masked faults: {100*masked/(masked + not_critical + critical)} %')
        print('not critical faults:', not_critical)
        print(f'% not critical: {100*not_critical/(masked + not_critical + critical)} %')
        print('SDC-1:', critical)   
        print(f'% critical: {100*critical/(masked + not_critical + critical)} %')
        
        # statistics
        total_outputs = masked + not_critical + critical
        percent_masked = 100 * masked / total_outputs
        percent_not_critical = 100 * not_critical / total_outputs
        percent_critical = 100 * critical / total_outputs
        
        with open(f'{SETTINGS.FI_ANALYSIS_PATH}/fault_statistics.txt', 'w') as file:
            file.write(f'total outputs: {total_outputs}\n')
            file.write(f'masked: {masked}\n')
            file.write(f'% masked faults: {percent_masked} %\n')
            file.write(f'not critical faults: {not_critical}\n')
            file.write(f'% not critical: {percent_not_critical} %\n')
            file.write(f'SDC-1: {critical}\n')
            file.write(f'% critical: {percent_critical} %\n')
       

    return output_results_list


def fault_list_gen():
    # Set a seed for reproducibility
    random_seed = SETTINGS.SEED  # You can choose any integer as the seed


    PRINT = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # network = SETTINGS.fault_list_model
    network = get_network(network_name=SETTINGS.NETWORK_NAME,
                        device=device,
                        dataset_name=SETTINGS.DATASET_NAME
                        )

    network.to(device)
    dataset_name = SETTINGS.DATASET_NAME
    network_name = SETTINGS.NETWORK_NAME

    # if dataset_name == 'CIFAR10' and 'ResNet' in network_name:
    #     network_path = SETTINGS.MODEL_TH_PATH
    # elif dataset_name == 'CIFAR10' or 'GTSRB':
    #     network_path = SETTINGS.MODEL_PT_PATH
    # elif dataset_name == 'CIFAR100':
    #     network_path = SETTINGS.MODEL_PTH_PATH
        

    # function = None
    # network.to(device)
    # state_dict = torch.load(network_path, map_location=device)['state_dict'] if '.th' in network_path else torch.load(network_path, map_location=device)
    # clean_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()} if function is None else {key.replace('module.', ''): function(value) if not (('bn' in key) and ('weight' in key)) else value for key, value in state_dict.items()}
    # network.load_state_dict(clean_state_dict, strict=False)
    # network.eval()

    feature_maps_layer_names = [name.replace('.weight', '') for name, module in network.named_modules()
                                            if isinstance(module, SETTINGS.modules_to_fault)]

    print(feature_maps_layer_names)
            
    total_sum_params = 0

    layer_params_list = []

    for name, param in network.named_parameters():
        if len(param.size()) >= 2:
            total_sum_params += param.numel()
            total_params = param.numel()
            layer_params_list.append((name, total_params))
            

    # -----------------------------------------------------------------------------------------------

    # LIST OF LAYER NAMES AND TOTAL PARAMETERS
    if PRINT:
        print("\nList of Layer Names and Total Parameters:")
    for layer_name, layer_param in layer_params_list:
        if PRINT:
            print(f"Layer: {layer_name}, Total Parameters: {layer_param}")
            total_params = total_params + layer_param

    print(f"total params: {total_params}") 
        
    p = SETTINGS.probability
    e = SETTINGS.error_margin
    t = SETTINGS.confidence_constant
    N = total_sum_params*SETTINGS.bit*2

    print(f"total faults: {N}")

    fault_to_inject = round(N/(1+e**2*(N-1)/(t**2*p*(1-p))))

    print(f"fault to inject: {fault_to_inject}")

    faults_to_inject_list = []

    for layer_name, total_params in layer_params_list:
        y = round((total_params * fault_to_inject) / total_sum_params)
        faults_to_inject_list.append((layer_name, y))

    # -----------------------------------------------------------------------------------------------

    # LIST OF FAULTS TO INJECT FOR EACH LAYER
    print("\nList of Faults to Inject for Each Layer:")
    for layer_name, faults_to_inject in faults_to_inject_list:
        if PRINT:
            print(f"Layer: {layer_name}, Faults to Inject: {faults_to_inject}")
        

    import numpy as np

    # Create a list or dictionary to store layer dimensions as NumPy arrays
    layer_dimensions_list = []

    # Iterate through the named parameters and save the dimensions
    for layer_name, parameters in network.named_parameters():
        if len(parameters.size()) >= 2:
            layer_dimensions = parameters.size()
            layer_dimensions_np = np.array(layer_dimensions)
            layer_dimensions_list.append((layer_name, layer_dimensions_np))

    # -----------------------------------------------------------------------------------------------

    # LIST OF LAYER NAMES AND DIMENSIONS
    print("\nList of Layer Names and Dimensions:")
    for layer_name, dimensions in layer_dimensions_list:
        if PRINT:
            print(f"Layer: {layer_name}, Dimensions: {dimensions}")

    import csv
    import random



    random.seed(random_seed)

    os.makedirs(SETTINGS.FAULT_LIST_PATH, exist_ok=True)
    csv_filename = f'{SETTINGS.FAULT_LIST_PATH}/{SETTINGS.FAULT_LIST_NAME}'
    header = ['Injection', 'Layer', 'TensorIndex', 'Bit']
    counter = 0
    with open(csv_filename, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header to the CSV file
        csv_writer.writerow(header)

        # Use a set to keep track of generated indices
        used_indices = set()
        # Iterate through each fault to inject for each layer
        row_number = 0
        max_attempts = 1000  # Adjust as needed
        for injection_number, (layer_name, total_params) in enumerate(layer_params_list):
            layer_dimensions = next(dimensions for name, dimensions in layer_dimensions_list if name == layer_name)
            for _ in range(faults_to_inject_list[injection_number][1]):
                
                attempts = 0
                while attempts < max_attempts:
                    if len(layer_dimensions) == 4:
                        # Generate random indices within the layer dimensions
                        height_index = random.randint(0, layer_dimensions[0] - 1)
                        width_index = random.randint(0, layer_dimensions[1] - 1)
                        depth_index = random.randint(0, layer_dimensions[2] - 1)
                        channel_index = random.randint(0, layer_dimensions[3] - 1)
                        tensor_index = f'({height_index}, {width_index}, {depth_index}, {channel_index})'
                    elif len(layer_dimensions) == 3:
                        height_index = random.randint(0, layer_dimensions[0] - 1)
                        width_index = random.randint(0, layer_dimensions[1] - 1)
                        depth_index = random.randint(0, layer_dimensions[2] - 1)
                        tensor_index = f'({height_index}, {width_index}, {depth_index})'
                    elif len(layer_dimensions) == 2:
                        height_index = random.randint(0, layer_dimensions[0] - 1)
                        width_index = random.randint(0, layer_dimensions[1] - 1)
                        tensor_index = f'({height_index}, {width_index})'

                    bit_flip = random.randint(0, 31)
                    
                    layer_name_no_weight = layer_name.replace('.weight', '')

                    # Check if the combination of layer_name, tensor_index, and bit_flip is already used
                    index_key = (layer_name_no_weight, tensor_index, bit_flip)
                    if index_key not in used_indices:
                        break  # Break the loop if the combination is unique
                    else:
                        attempts += 1
                        counter = counter + 1

                if attempts == max_attempts:
                    print(f"Could not find a unique combination for {layer_name}. Increase max_attempts if needed.")
                    break

                # Add the combination to the set of used indices
                used_indices.add(index_key)

                # Write the data to the CSV file
                csv_writer.writerow([row_number, layer_name_no_weight, tensor_index, bit_flip])
                row_number += 1

    print(f"Number of attempted indices: {counter}")
    print(f"Number of duplicate indices: {row_number - len(used_indices)}")
    print(f"Number of unique indices: {len(used_indices)}")
    print(f"CSV file '{csv_filename}' has been created successfully.")




        