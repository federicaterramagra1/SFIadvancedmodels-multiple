import os

import numpy as np
import random
import pandas as pd
import shutil
from typing import Union, List, Tuple
from functools import reduce

import SETTINGS
import torch
from torch.nn import Sequential, Module

from torch.utils.data import DataLoader, TensorDataset

from faultManager.FaultListManager import FLManager
from faultManager.NeuronFault import NeuronFault
from faultManager.WeightFault import WeightFault

from dlModels.BreastCancer.mlp import SimpleMLP

from torchvision.models import resnet
from torchvision.models.densenet import _DenseBlock, _Transition
from torchvision.models.efficientnet import Conv2dNormActivation
from torchvision import transforms
from torchvision.datasets import GTSRB, CIFAR10, CIFAR100, MNIST, ImageNet
from torchvision.transforms.v2 import ToTensor,Resize,Compose,ColorJitter,RandomRotation,AugMix,GaussianBlur,RandomEqualize,RandomHorizontalFlip,RandomVerticalFlip

import csv
from tqdm import tqdm

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class UnknownNetworkException(Exception):
    pass


def clean_inference(network, loader, device, network_name):
    clean_output_scores = list()
    clean_output_indices = list()
    clean_labels = list()

    counter = 0
    dataset_size = 0

    with torch.no_grad():
        pbar = tqdm(loader, colour='green', desc='Clean Run')

        for batch_id, batch in enumerate(pbar):
            data, label = batch
            dataset_size += len(label)
            data = data.to(device)
            
            network_output = network(data)
            prediction = torch.argmax(network_output, dim=1)
            scores = network_output.cpu()
            indices = prediction.cpu().tolist()

            clean_output_scores.append(scores)
            clean_output_indices.append(indices)
            clean_labels.append(label)

            counter += 1

        elementwise_comparison = [
            label != index 
            for labels, indices in zip(clean_labels, clean_output_indices) 
            for label, index in zip(labels, indices)
        ]

        # Count the number of different elements
        num_different_elements = elementwise_comparison.count(True)
        
        print(f'device: {device}')
        print(f'network: {network_name}')
        print(f"The DNN wrong predictions are: {num_different_elements}")
        accuracy = (1 - num_different_elements / dataset_size) * 100
        print(f"The final accuracy is: {accuracy:.2f}%")

def faulty_inference(network, loader, device, network_name, faults_injected=False):
    faulty_output_indices = list()
    true_labels = list()
    total = 0
    wrong = 0

    with torch.no_grad():
        pbar = tqdm(loader, colour='red', desc='Faulty Run')

        for batch_id, (data, label) in enumerate(pbar):
            data = data.to(device)
            label = label.to(device)

            output = network(data)
            pred = torch.argmax(output, dim=1)

            total += len(label)
            wrong += (pred != label).sum().item()

            faulty_output_indices.append(pred.cpu())
            true_labels.append(label.cpu())

    accuracy = (1 - wrong / total) * 100
    print(f"\nFaulty inference results on device: {device}")
    print(f"Model: {network_name}")
    print(f"Wrong predictions: {wrong}")
    print(f"Faulty model accuracy: {accuracy:.2f}%")


def train_model(model, train_loader, val_loader=None, num_epochs=50, lr=0.001, device='cpu', save_path=None):
    print(f"\n Inizio training su {device} per {num_epochs} epoche...")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        acc = 100 * correct / total
        print(f" Epoch {epoch+1}/{num_epochs} | Loss: {total_loss:.4f} | Train Accuracy: {acc:.2f}%")

        # Valutazione su validation set
        if val_loader:
            model.eval()
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    correct_val += (predicted == labels).sum().item()
                    total_val += labels.size(0)
            val_acc = 100 * correct_val / total_val
            print(f" Validation Accuracy: {val_acc:.2f}%")

            # Salva il miglior modello
            if save_path and val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), save_path)
                print(f" Salvataggio modello migliore in {save_path}")

    print(" Training completato.")
    return model

def get_network(network_name: str,
                device: torch.device,
                dataset_name: str,
                root: str = '.') -> torch.nn.Module:
    
    # Load the network by using the name of the mode and the dataset
    if dataset_name == 'BreastCancer':
        print(f'Loading network {network_name} for BreastCancer ...')
        if network_name == 'SimpleMLP':
            from dlModels.BreastCancer.mlp import SimpleMLP
            network = SimpleMLP()
            # Explicitly attach the quantize method to the network
            network.quantize_model = network.quantize_model
            # Wrap the model for quantization
            network = torch.quantization.QuantWrapper(network)
            network.qconfig = torch.quantization.get_default_qconfig("fbgemm")  # Suitable for x86 CPUs
            # Explicitly attach the quantize method to the wrapped model
            network.quantize_model = network.module.quantize_model
        elif network_name == 'BiggerMLP':
            from dlModels.BreastCancer.bigger_mlp import BiggerMLP
            network = BiggerMLP()
            # Explicitly attach the quantize method to the network
            network.quantize_model = network.quantize_model
            # Wrap the model for quantization
            network = torch.quantization.QuantWrapper(network)
            network.qconfig = torch.quantization.get_default_qconfig("fbgemm")  # Suitable for x86 CPUs
            # Explicitly attach the quantize method to the wrapped model
            network.quantize_model = network.module.quantize_model
        else:
            raise ValueError(f"Unknown network '{network_name}' for dataset '{dataset_name}'")
    elif dataset_name == 'Banknote':
            from dlModels.Banknote.mlp import SimpleMLP
            print(f'Loading network {network_name} for Banknote ...')
            if network_name == 'SimpleMLP':
                from dlModels.Banknote.mlp import SimpleMLP
                network = SimpleMLP()
                network.to(device)

            else:
                raise ValueError(f"Unknown network '{network_name}' for dataset '{dataset_name}'")
    # Move the model to the specified device
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
    if dataset_name == 'Banknote':
        from utils import load_banknote_dataset
        print('Loading Banknote dataset...')
        train_loader, val_loader, test_loader = load_banknote_dataset(batch_size=batch_size)
        return train_loader, val_loader, test_loader

        
    if network_name == 'SimpleMLP':
        # Load Breast Cancer dataset with the correct parameters
        train_loader, val_loader, test_loader = load_breastCancer_datasets(
            train_batch_size=batch_size,
            test_batch_size=batch_size
        )
        return train_loader, test_loader  # Return only train and test loaders

    if network_name == 'BiggerMLP':
        # Load Breast Cancer dataset with the correct parameters
        train_loader, val_loader, test_loader = load_breastCancer_datasets(
            train_batch_size=batch_size,
            test_batch_size=batch_size
        )
        return train_loader, test_loader  # Return only train and test loaders

            
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
    elif dataset_name == 'BreastCancer':
        print('Loading BreastCancer dataset...')
        train_loader, val_loader, test_loader = load_breastCancer_datasets(
            train_batch_size=batch_size,
            test_batch_size=batch_size
        )


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
                   fault_list_generator: FLManager,
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
    elif fault_model == 'bit-flip':
        fault_list = fault_list_generator.get_weight_fault_list()
    else:
        raise ValueError(f'Invalid fault model {fault_model}')


    injectable_modules = fault_list_generator.injectable_output_modules_list

    return fault_list, injectable_modules


def get_device(
               use_cuda0: bool,
               use_cuda1: bool) -> torch.device:
    """
    Get the device where to perform the fault injection
    :param forbid_cuda: Forbids the usage of cuda. Overrides use_cuda
    :param use_cuda: Whether to use the cuda device or the cpu
    :return: The device where to perform the fault injection
    """

    if use_cuda0:
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = ''
            print('ERROR: cuda:0 not available even if use-cuda is set')
            exit(-1)
    elif use_cuda1:
        if torch.cuda.is_available():
            device = 'cuda:1'
        else:
            device = ''
            print('ERROR: cuda:1 not available even if use-cuda is set')
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

def load_breastCancer_datasets(train_batch_size=32, train_split=0.8, test_batch_size=1, test_image_per_class=None):
    """
    Load the Breast Cancer dataset and split it into training, validation, and test sets.
    
    :param train_batch_size: Batch size for the training DataLoader.
    :param train_split: Ratio of the training set to use for training (the rest is used for validation).
    :param test_batch_size: Batch size for the test DataLoader.
    :param test_image_per_class: Not used for Breast Cancer dataset (included for consistency with other datasets).
    :return: Tuple of (train_loader, val_loader, test_loader).
    """
    # Load the breast cancer dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # Split the training set into training and validation
    train_split_length = int(len(train_dataset) * train_split)
    val_split_length = len(train_dataset) - train_split_length
    train_subset, val_subset = torch.utils.data.random_split(train_dataset,
                                                             lengths=[train_split_length, val_split_length],
                                                             generator=torch.Generator().manual_seed(1234))

    # Create DataLoader objects
    train_loader = DataLoader(dataset=train_subset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_subset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)

    print('Breast Cancer Dataset loaded')

    return train_loader, val_loader, test_loader

def load_banknote_dataset(batch_size=32):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    column_names = ["variance", "skewness", "curtosis", "entropy", "class"]
    df = pd.read_csv(url, header=None, names=column_names)

    X = df.drop("class", axis=1).values
    y = df["class"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split iniziale: 70% train, 30% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_scaled, y, train_size=0.7, random_state=42, stratify=y
    )

    # Split interno: 90% train, 10% val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1, random_state=42, stratify=y_trainval
    )

    # Conversione in tensori
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

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
    
    
import os
import numpy as np

def ensure_directory_exists(directory):
    """
    Ensure that the directory exists. If not, create it.
    """
    if not os.path.exists(directory):
        print(f"Creating missing directory: {directory}")
        os.makedirs(directory)

def ensure_file_exists(file_path, default_data=None):
    """
    Check if the file exists and create it with default_data if not.
    """
    if not os.path.exists(file_path):
        print(f"Creating missing file: {file_path}")
        # If you need to create a numpy file, for example:
        if default_data is not None:
            np.save(file_path, default_data)  # You can replace this with your own logic to save initial data
        else:
            # Create an empty file (if needed)
            open(file_path, 'w').close()

def count_batch(folder, path):
    try:
        # Ensure directory exists before listing files
        if not os.path.exists(folder):
            print(f"Error: Directory {folder} does not exist.")
            return 0, 0, 0  # Return default values to prevent crashes

        files = os.listdir(folder)
        if not files:
            print(f"Warning: No files found in {folder}.")
            return 0, 0, 0  # Handle empty directories gracefully

        loaded_file = np.load(path, allow_pickle=True)
        if loaded_file.size == 0:
            print(f"Warning: {path} is empty. Skipping.")
            return 0, 0, 0  # Skip if empty

        n_outputs = loaded_file.shape[2]
        n_faults = loaded_file.shape[0]
        return len(files), n_outputs, n_faults

    except FileNotFoundError:
        print(f"Error: File not found {path}.")
        return 0, 0, 0
    except EOFError:
        print(f"Error: No data left in file {path}. Skipping.")
        return 0, 0, 0
    except Exception as e:
        print(f"Unexpected error in count_batch: {e}")
        return 0, 0, 0


import os
import numpy as np
import pandas as pd
import shutil
import SETTINGS
from tqdm import tqdm
def output_definition(test_loader, batch_size):
    clean_output_path = os.path.join(SETTINGS.CLEAN_OUTPUT_FOLDER, 'clean_output.npy')
    if not os.path.exists(clean_output_path):
        print(f" ERROR: Clean output file {clean_output_path} is missing!")
        return

    print(" Caricamento output clean...")
    loaded_clean_output = np.load(clean_output_path, allow_pickle=True)
    number_of_clean_batches = len(loaded_clean_output)

    batch_folder = os.path.join(SETTINGS.FAULTY_OUTPUT_FOLDER, SETTINGS.FAULT_MODEL)
    batch_path = os.path.join(batch_folder, 'batch_0.npy')
    number_of_batch, _, max_faults = count_batch(batch_folder, batch_path)
    number_of_batch = min(number_of_batch, number_of_clean_batches)
    n_faults = max_faults if SETTINGS.FAULTS_TO_INJECT == -1 else SETTINGS.FAULTS_TO_INJECT

    output_path = os.path.join(SETTINGS.FI_ANALYSIS_PATH, "output_analysis.csv")
    os.makedirs(SETTINGS.FI_ANALYSIS_PATH, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Fault_ID', 'batch', 'image', 'output'])

        for fault_id in tqdm(range(n_faults), desc="Analysis", colour='cyan'):
            for i in range(number_of_batch):
                path = os.path.join(batch_folder, f'batch_{i}.npy')
                if not os.path.exists(path):
                    continue

                faulty = np.load(path, allow_pickle=True)
                if fault_id >= faulty.shape[0]:
                    continue

                clean_batch = loaded_clean_output[i]
                current_batch_size = min(len(clean_batch), batch_size)

                clean_top = np.argmax(clean_batch, axis=1)
                faulty_top = np.argmax(faulty[fault_id], axis=1)
                same_label = (clean_top == faulty_top)

                clean_conf = clean_batch[np.arange(current_batch_size), clean_top]
                faulty_conf = faulty[fault_id][np.arange(current_batch_size), clean_top]
                delta = np.abs(faulty_conf - clean_conf) / np.maximum(1e-8, np.abs(clean_conf))

                masked = np.all(clean_batch == faulty[fault_id], axis=1)

                outputs = np.full(current_batch_size, 4)
                outputs[masked] = 0
                outputs[same_label & (delta < 0.1)] = 1
                outputs[same_label & (delta >= 0.1) & (delta < 0.2)] = 2
                outputs[same_label & (delta >= 0.2)] = 3

                for j in range(current_batch_size):
                    writer.writerow([fault_id, i, j, str(outputs[j])])

    print("\nOutput analysis completata.")

from concurrent.futures import ProcessPoolExecutor


def _analyze_fault_range(start_id, end_id, batch_folder, clean_output_path, batch_size, n_batches):
    results = []

    clean_output = np.load(clean_output_path, allow_pickle=True)

    for fault_id in range(start_id, end_id):
        for i in range(n_batches):
            path = os.path.join(batch_folder, f'batch_{i}.npy')
            if not os.path.exists(path):
                continue

            faulty = np.load(path, allow_pickle=True)
            if fault_id >= faulty.shape[0]:
                continue

            clean_batch = clean_output[i]
            for j in range(min(len(clean_batch), batch_size)):
                clean = clean_batch[j]
                faulty_out = faulty[fault_id, j]
                clean_top = np.argmax(clean)
                faulty_top = np.argmax(faulty_out)
                delta = abs(faulty_out[clean_top] - clean[clean_top]) / max(1e-8, abs(clean[clean_top]))

                if np.array_equal(clean, faulty_out):
                    results.append([fault_id, i, j, '0'])
                elif clean_top == faulty_top:
                    if delta >= 0.2:
                        results.append([fault_id, i, j, '3'])
                    elif delta >= 0.1:
                        results.append([fault_id, i, j, '2'])
                    else:
                        results.append([fault_id, i, j, '1'])
                else:
                    results.append([fault_id, i, j, '4'])
    return results

def _analyze_fault_range_star(args):
    return _analyze_fault_range(*args)


from concurrent.futures import ProcessPoolExecutor, as_completed

def output_definition_parallel(test_loader, batch_size, n_workers=32):
    import math

    clean_output_path = os.path.join(SETTINGS.CLEAN_OUTPUT_FOLDER, 'clean_output.npy')
    if not os.path.exists(clean_output_path):
        print(f" ERROR: Clean output file {clean_output_path} is missing!")
        return

    batch_folder = os.path.join(SETTINGS.FAULTY_OUTPUT_FOLDER, SETTINGS.FAULT_MODEL)
    batch_path = os.path.join(batch_folder, 'batch_0.npy')
    number_of_batch, _, max_faults = count_batch(batch_folder, batch_path)

    loaded_clean_output = np.load(clean_output_path, allow_pickle=True)
    number_of_batch = min(number_of_batch, len(loaded_clean_output))
    n_faults = max_faults if SETTINGS.FAULTS_TO_INJECT == -1 else SETTINGS.FAULTS_TO_INJECT

    print(f" Parallel analysis with {n_workers} workers on {n_faults} faults × {number_of_batch} batches")

    os.makedirs(SETTINGS.FI_ANALYSIS_PATH, exist_ok=True)
    output_path = os.path.join(SETTINGS.FI_ANALYSIS_PATH, "output_analysis.csv")

    chunk_size = math.ceil(n_faults / n_workers)
    chunks = [(i, min(i + chunk_size, n_faults)) for i in range(0, n_faults, chunk_size)]
    args = [(start, end, batch_folder, clean_output_path, batch_size, number_of_batch) for (start, end) in chunks]

    all_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_analyze_fault_range_star, arg) for arg in args]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Parallel Fault Analysis"):
            all_results.extend(future.result())

    print(f" Saving analysis to {output_path}...")
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Fault_ID', 'batch', 'image', 'output'])
        writer.writerows(all_results)

    print(" Output analysis completata e salvata.")



def _init_clean_output(path):
    global CLEAN_OUTPUT
    CLEAN_OUTPUT = np.load(path, allow_pickle=True)

def _analyze_fault_range_chunk(chunk_id, fault_ids, batch_folder, batch_size, n_batches, output_dir):
    global CLEAN_OUTPUT
    results = []

    print(f"[Chunk {chunk_id}] Analisi fault IDs: {fault_ids[:5]}...")  # per conferma

    for fault_id in fault_ids:
        for i in range(n_batches):
            path = os.path.join(batch_folder, f'batch_{i}.npy')
            if not os.path.exists(path):
                continue

            faulty = np.load(path, allow_pickle=True, mmap_mode='r')
            if fault_id >= faulty.shape[0]:
                continue

            clean_batch = CLEAN_OUTPUT[i]
            for j in range(min(len(clean_batch), batch_size)):
                clean = clean_batch[j]
                faulty_out = faulty[fault_id, j]
                clean_top = np.argmax(clean)
                faulty_top = np.argmax(faulty_out)
                delta = abs(faulty_out[clean_top] - clean[clean_top]) / max(1e-8, abs(clean[clean_top]))

                if np.array_equal(clean, faulty_out):
                    results.append([fault_id, i, j, '0'])
                elif clean_top == faulty_top:
                    if delta >= 0.2:
                        results.append([fault_id, i, j, '3'])
                    elif delta >= 0.1:
                        results.append([fault_id, i, j, '2'])
                    else:
                        results.append([fault_id, i, j, '1'])
                else:
                    results.append([fault_id, i, j, '4'])
    

    chunk_file = os.path.join(output_dir, f"output_analysis_part_{chunk_id}.csv")
    with open(chunk_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Fault_ID', 'batch', 'image', 'output'])
        writer.writerows(results)

    print(f"[Chunk {chunk_id}] Completato. Salvato in {chunk_file}")


    return chunk_file

def _analyze_fault_range_chunk_star(args):
    return _analyze_fault_range_chunk(*args)

def output_definition_parallel_chunked(test_loader, batch_size, n_workers=32, chunk_size=5000):
    from utils import count_batch

    clean_output_path = os.path.join(SETTINGS.CLEAN_OUTPUT_FOLDER, 'clean_output.npy')
    if not os.path.exists(clean_output_path):
        print(f" Clean output not found: {clean_output_path}")
        return

    print(f"Loaded CLEAN_OUTPUT with shape: {np.load(clean_output_path, allow_pickle=True).shape}")

    batch_folder = os.path.join(SETTINGS.FAULTY_OUTPUT_FOLDER, SETTINGS.FAULT_MODEL)
    batch_path = os.path.join(batch_folder, 'batch_0.npy')
    number_of_batch, _, max_faults = count_batch(batch_folder, batch_path)

    n_faults = max_faults if SETTINGS.FAULTS_TO_INJECT == -1 else SETTINGS.FAULTS_TO_INJECT
    output_dir = SETTINGS.FI_ANALYSIS_PATH
    os.makedirs(output_dir, exist_ok=True)

    fault_ids = list(range(n_faults))
    chunks = [fault_ids[i:i + chunk_size] for i in range(0, n_faults, chunk_size)]

    args = []
    for chunk_id, fault_chunk in enumerate(chunks):
        chunk_file = os.path.join(output_dir, f"output_analysis_part_{chunk_id}.csv")
        if os.path.exists(chunk_file):
            print(f" Chunk {chunk_id} already exists, skipping.")
            continue
        args.append((chunk_id, fault_chunk, batch_folder, batch_size, number_of_batch, output_dir))

    print(f" Avvio analisi parallela in {len(args)} chunk da {chunk_size} fault, usando {n_workers} core...")

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_clean_output, initargs=(clean_output_path,)) as executor:
        list(tqdm(executor.map(_analyze_fault_range_chunk_star, args), total=len(args), desc="Parallel Fault Analysis"))


    # Merge finale
    merged_path = os.path.join(output_dir, "output_analysis.csv")
    with open(merged_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Fault_ID', 'batch', 'image', 'output'])
        for chunk_id in range(len(chunks)):
            chunk_file = os.path.join(output_dir, f"output_analysis_part_{chunk_id}.csv")
            if os.path.exists(chunk_file):
                with open(chunk_file, 'r') as infile:
                    next(infile)  # Skip header
                    for row in infile:
                        outfile.write(row)
    print(f" Analisi completata. File finale salvato in {merged_path}")

    
def csv_summary():
    FI_ANALYSIS_PATH = SETTINGS.FI_ANALYSIS_PATH
    FAULT_LIST_PATH = f'{SETTINGS.FAULT_LIST_PATH}/{SETTINGS.FAULT_LIST_NAME}'
    OUTPUT_FILE_PATH = SETTINGS.FI_SUM_ANALYSIS_PATH

    print(f"FI_ANALYSIS_PATH: {FI_ANALYSIS_PATH}")
    print(f"Fault list path: {FAULT_LIST_PATH}")
    print(f"Summary output path: {OUTPUT_FILE_PATH}")

    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)

    df_out = pd.read_csv(f'{FI_ANALYSIS_PATH}/output_analysis.csv')
    print(' Output analysis loaded.')
    df_fault = pd.read_csv(FAULT_LIST_PATH)
    print(' Fault list loaded.')

    injection_ids = df_fault['Injection'].unique()

    print(f" Starting multiprocessing with {cpu_count()} cores...")

    with Pool(processes=cpu_count(), initializer=init_worker, initargs=(df_out, df_fault)) as pool:
        results = list(tqdm(pool.imap(process_injection, injection_ids), total=len(injection_ids), desc="Calculating summary"))

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f" Summary CSV saved to {OUTPUT_FILE_PATH}")



from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def _process_injection(inj_id, group_df, df):
    fault_outputs = df[df['Fault_ID'] == inj_id]['output']

    masked = (fault_outputs == 0).sum()
    non_critical = fault_outputs.isin([1, 2, 3]).sum()
    critical = (fault_outputs == 4).sum()

    layers = group_df['Layer'].tolist()
    indices = group_df['TensorIndex'].tolist()
    bits = group_df['Bit'].tolist()

    total = masked + non_critical + critical
    accuracy = (masked + non_critical) / total if total > 0 else 0.0
    failure_rate = critical / total if total > 0 else 0.0

    return {
        'Injection': inj_id,
        'Layers': str(layers),
        'TensorIndices': str(indices),
        'Bits': str(bits),
        'masked': masked,
        'non_critical': non_critical,
        'critical': critical,
        'accuracy': round(accuracy, 4),
        'failure_rate': round(failure_rate, 4)
    }

import pandas as pd
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def _process_injection_star(args):
    inj_id, group_df, df = args
    return _process_injection(inj_id, group_df, df)

def csv_summary_parallel(n_workers=32):
    input_file_path = f'{SETTINGS.FI_ANALYSIS_PATH}/output_analysis.csv'
    fault_list_path = f'{SETTINGS.FAULT_LIST_PATH}/{SETTINGS.FAULT_LIST_NAME}'
    output_file_path = f'{SETTINGS.FI_SUM_ANALYSIS_PATH}'

    print(f"FI_ANALYSIS_PATH: {SETTINGS.FI_ANALYSIS_PATH}")
    print(f"Fault list path: {SETTINGS.FAULT_LIST_PATH}")
    print(f"Summary output path: {output_file_path}")

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    try:
        df = pd.read_csv(input_file_path)
        print(' Output analysis loaded.')
    except FileNotFoundError:
        print(f" File not found: {input_file_path}")
        return

    try:
        fault_df = pd.read_csv(fault_list_path)
        print(' Fault list loaded.')
    except FileNotFoundError:
        print(f" File not found: {fault_list_path}")
        return

    grouped = fault_df.groupby("Injection")
    args = [(inj_id, group.copy(), df) for inj_id, group in grouped]

    print(f" Parallelizing summary on {n_workers} cores...")
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        output_rows = list(tqdm(executor.map(_process_injection_star, args),
                                total=len(args),
                                desc="Calculating summary"))

    summary_df = pd.DataFrame(output_rows)
    summary_df.to_csv(output_file_path, index=False)
    print(f" Summary CSV saved to {output_file_path}")


def _process_summary_chunk(chunk_id, injection_ids, fault_df_path, analysis_csv_path, output_dir):
    df = pd.read_csv(analysis_csv_path)
    fault_df = pd.read_csv(fault_df_path)

    fault_outputs_dict = df.groupby("Fault_ID")["output"].apply(list).to_dict()
    grouped = fault_df.groupby("Injection")

    output_rows = []
    for inj_id in injection_ids:
        if inj_id not in grouped.groups:
            continue
        group_df = grouped.get_group(inj_id)
        outputs = fault_outputs_dict.get(inj_id, [])

        masked = sum(1 for o in outputs if o == 0)
        non_critical = sum(1 for o in outputs if o in [1, 2, 3])
        critical = sum(1 for o in outputs if o == 4)

        layers = group_df['Layer'].tolist()
        indices = group_df['TensorIndex'].tolist()
        bits = group_df['Bit'].tolist()

        total = masked + non_critical + critical
        accuracy = (masked + non_critical) / total if total > 0 else 0.0
        failure_rate = critical / total if total > 0 else 0.0

        output_rows.append({
            'Injection': inj_id,
            'Layers': str(layers),
            'TensorIndices': str(indices),
            'Bits': str(bits),
            'masked': masked,
            'non_critical': non_critical,
            'critical': critical,
            'accuracy': round(accuracy, 4),
            'failure_rate': round(failure_rate, 4)
        })

    chunk_path = os.path.join(output_dir, f"summary_chunk_{chunk_id}.csv")
    pd.DataFrame(output_rows).to_csv(chunk_path, index=False)
    return chunk_path

def _process_summary_chunk_star(args):
    return _process_summary_chunk(*args)

def csv_summary_parallel_chunked(n_workers=32, chunk_size=5000):
    import SETTINGS

    analysis_csv_path = os.path.join(SETTINGS.FI_ANALYSIS_PATH, "output_analysis.csv")
    fault_df_path = os.path.join(SETTINGS.FAULT_LIST_PATH, SETTINGS.FAULT_LIST_NAME)
    final_output_path = SETTINGS.FI_SUM_ANALYSIS_PATH
    output_dir = os.path.dirname(final_output_path)

    os.makedirs(output_dir, exist_ok=True)

    try:
        fault_df = pd.read_csv(fault_df_path)
        print("✅ Fault list loaded.")
    except FileNotFoundError:
        print(f"❌ File not found: {fault_df_path}")
        return

    injection_ids = fault_df["Injection"].unique()
    chunks = [injection_ids[i:i + chunk_size] for i in range(0, len(injection_ids), chunk_size)]
    args = [(i, list(chunk), fault_df_path, analysis_csv_path, output_dir) for i, chunk in enumerate(chunks)]

    print(f"🧠 Avvio sintesi CSV su {len(chunks)} chunk da {chunk_size} con {n_workers} worker...")
    chunk_files = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for chunk_path in tqdm(executor.map(_process_summary_chunk_star, args),
                               total=len(args), desc="Calculating chunked summary"):
            chunk_files.append(chunk_path)

    # Fusione finale
    summary_df = pd.concat([pd.read_csv(f) for f in chunk_files], ignore_index=True)
    summary_df.to_csv(final_output_path, index=False)
    print(f"✅ Summary CSV finale salvato in {final_output_path}")


import numpy as np
import random


import itertools
import csv
import os

def fault_list_gen():

    random.seed(SETTINGS.SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = get_network(network_name=SETTINGS.NETWORK_NAME,
                          device=device,
                          dataset_name=SETTINGS.DATASET_NAME)
    network.to(device)

    bit_width = 8  # 8-bit quantization
    bit_positions = list(range(bit_width))
    all_bit_faults = []  # contiene tutte le tuple (layer, index, bit)

    # 1. Estrai tutti i singoli bit flip possibili e salvali in un CSV intermedio
    intermediate_singles_path = f"{SETTINGS.FAULT_LIST_PATH}/single_faults.csv"
    os.makedirs(SETTINGS.FAULT_LIST_PATH, exist_ok=True)
    with open(intermediate_singles_path, 'w', newline='') as single_file:
        writer = csv.writer(single_file)
        writer.writerow(['Injection', 'Layer', 'TensorIndex', 'Bit'])
        inj_id = 0
        for name, param in network.named_parameters():
            if 'weight' in name:
                layer_name = name.replace('.weight', '')
                shape = param.shape
                for idx in itertools.product(*[range(s) for s in shape]):
                    for bit in bit_positions:
                        all_bit_faults.append((layer_name, idx, bit))
                        writer.writerow([inj_id, layer_name, idx, bit])
                        inj_id += 1

    print(f" Salvati {len(all_bit_faults)} bit flip singoli in {intermediate_singles_path}")

    # 2. Genera tutte le combinazioni da NUM_FAULTS_TO_INJECT e salvale in CSV intermedio
    intermediate_combos_path = f"{SETTINGS.FAULT_LIST_PATH}/combinations_{SETTINGS.NUM_FAULTS_TO_INJECT}.csv"
    combinations = list(itertools.combinations(all_bit_faults, SETTINGS.NUM_FAULTS_TO_INJECT))

    with open(intermediate_combos_path, 'w', newline='') as combo_file:
        writer = csv.writer(combo_file)
        writer.writerow(['GroupID'] + [f"Fault{i+1}" for i in range(SETTINGS.NUM_FAULTS_TO_INJECT)])
        for i, combo in enumerate(combinations):
            row = [i] + [f"{layer},{idx},{bit}" for (layer, idx, bit) in combo]
            writer.writerow(row)

    print(f" Salvate {len(combinations)} combinazioni di {SETTINGS.NUM_FAULTS_TO_INJECT} bit in {intermediate_combos_path}")

    # 3. Seleziona combinazioni da salvare: tutte o campione random
    if SETTINGS.FAULTS_TO_INJECT == -1:
        selected = combinations
        print(f" Fault list ESAUSTIVA: {len(selected)} combinazioni da {SETTINGS.NUM_FAULTS_TO_INJECT} bit flip.")
    else:
        selected = random.sample(combinations, SETTINGS.FAULTS_TO_INJECT)
        print(f" Fault list RANDOM: {SETTINGS.FAULTS_TO_INJECT} combinazioni da {SETTINGS.NUM_FAULTS_TO_INJECT} bit flip.")

    # 4. Scrivi il file finale
    final_csv_path = f"{SETTINGS.FAULT_LIST_PATH}/{SETTINGS.FAULT_LIST_NAME}"
    with open(final_csv_path, 'w', newline='') as final_file:
        writer = csv.writer(final_file)
        writer.writerow(['Injection', 'Layer', 'TensorIndex', 'Bit'])
        for inj_id, combo in enumerate(selected):
            for layer, idx, bit in combo:
                writer.writerow([inj_id, layer, idx, bit])

    print(f" Fault list finale scritta in {final_csv_path} con {len(selected)} iniezioni.")


def num_experiments_needed(p_estimate=0.5):
    e = SETTINGS.error_margin
    t = SETTINGS.confidence_constant
    n_exp = int((t ** 2 * p_estimate * (1 - p_estimate)) / (e ** 2))
    print(f"Numero minimo di esperimenti necessari: {n_exp}")
    return n_exp

import pandas as pd

def select_random_faults(fault_list_path, num_faults_needed):
    fault_df = pd.read_csv(fault_list_path)
    sampled_faults = fault_df.sample(n=num_faults_needed, random_state=SETTINGS.SEED)
    sampled_faults.to_csv(fault_list_path.replace('.csv', '_sampled.csv'), index=False)
    print(f"Fault casualmente selezionati salvati in {fault_list_path.replace('.csv', '_sampled.csv')}")