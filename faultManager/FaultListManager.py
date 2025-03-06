import os
import csv
import numpy as np
from tqdm import tqdm
from ast import literal_eval as make_tuple
from typing import Type, List
from torch.nn import Module
import torch
import torchinfo
import SETTINGS
from faultManager.WeightFault import WeightFault
from faultManager.NeuronFault import NeuronFault
from faultManager.modules.InjectableOutputModule import injectable_output_module_class

class FLManager:

    def __init__(self, network: Module, network_name: str, device: torch.device,
                 module_class: Type[Module] = None, input_size: torch.Size = None, save_ifm: bool = False):

        self.network = network
        self.network_name = network_name
        self.device = device

        # The class of the injectable modules
        self.module_class = module_class
        self.injectable_module_class = injectable_output_module_class(self.module_class)

        # List of injectable modules. Used only for neurons injection
        self.injectable_output_modules_list = None

        # Create the list of injectable module if the module_class is set
        if self.module_class is not None:
            self.__replace_injectable_output_modules(input_size=input_size, save_ifm=save_ifm)

        # Name of the injectable layers
        injectable_layer_names = [name.replace('.weight', '') for name, module in self.network.named_modules()
                                  if isinstance(module, self.module_class)]

        # List of the shape of all the layers that contain weight
        self.net_layer_shape = {name.replace('.weight', ''): param.shape for name, param in self.network.named_parameters()
                                if name.replace('.weight', '') in injectable_layer_names}

        # The fault list
        self.fault_list = None

    def __replace_injectable_output_modules(self, input_size: torch.Size, save_ifm: bool = False) -> None:
        """
        Replace the target modules with a version that is injectable.
        """
        modules_to_replace = [(name, module) for name, module in self.network.named_modules()
                              if isinstance(module, self.module_class)]

        # Initialize the list of all the injectable layers
        self.injectable_output_modules_list = list()

        # Create a summary of the network
        summary = torchinfo.summary(self.network, device=self.device, input_size=input_size, verbose=False)

        # Extract the output, input and kernel shape of all the convolutional layers of the network
        output_shapes = [torch.Size(info.output_size) for info in summary.summary_list
                         if isinstance(info.module, self.module_class)]
        input_shapes = [torch.Size(info.input_size) for info in summary.summary_list
                        if isinstance(info.module, self.module_class)]
        kernel_shapes = [info.module.weight.shape for info in summary.summary_list
                         if isinstance(info.module, self.module_class)]

        # Replace all layers with injectable convolutional layers
        for layer_id, (layer_name, layer_module) in enumerate(modules_to_replace):
            layer_module.__class__ = self.injectable_module_class
            layer_module.init_as_copy(device=self.device, layer_name=layer_name,
                                      input_shape=input_shapes[layer_id],
                                      output_shape=output_shapes[layer_id],
                                      kernel_shape=kernel_shapes[layer_id],
                                      save_ifm=save_ifm)

            # Append the layer to the list
            self.injectable_output_modules_list.append(layer_module)

    def update_network(self, network):
        self.network = network
        self.injectable_output_modules_list = [module for module in self.network.modules()
                                               if isinstance(module, self.injectable_module_class)]

    def get_neuron_fault_list(self):
        """
        Generate a fault list for the neurons according to the DATE09 formula.
        """
        cwd = os.getcwd()

        try:
            with open(f'{SETTINGS.FAULT_LIST_PATH}/{SETTINGS.FAULT_LIST_NAME}', newline='') as f_list:
                reader = csv.reader(f_list)
                fault_list = list(reader)[1:]
                fault_list = [NeuronFault(layer_name=str(fault[1]),
                                          layer_index=int(fault[2]),
                                          feature_map_index=make_tuple(fault[3]),
                                          value=float(fault[-1])) for fault in fault_list]

            print('Fault list loaded from file')

        except FileNotFoundError:
            print('Fault list not found')
            exit(-1)

    def get_weight_fault_list(self) -> List[WeightFault]:
        """
        Get the fault list for the weights, ensuring valid indices.
        """
        cwd = os.getcwd()

        try:
            with open(f'{SETTINGS.FAULT_LIST_PATH}/{SETTINGS.FAULT_LIST_NAME}', newline='') as f_list:
                reader = csv.reader(f_list)
                fault_list = list(reader)[1:]
                fault_list = [WeightFault(injection=int(fault[0]),
                                          layer_name=fault[1],
                                          tensor_index=make_tuple(fault[2]),
                                          bit=int(fault[-1])) for fault in fault_list]

            print('Fault list loaded from file')
            # Validate the fault list
            valid_fault_list = self.validate_fault_list(fault_list)
            self.fault_list = valid_fault_list
            return valid_fault_list

        except FileNotFoundError:
            print(f'Fault list not found: {SETTINGS.FAULT_LIST_PATH}/{SETTINGS.FAULT_LIST_NAME}')
            exit(-1)

    def validate_fault_list(self, fault_list: List[WeightFault]) -> List[WeightFault]:
        """
        Validate the fault list to ensure indices are within valid ranges.
        """
        valid_fault_list = []
        state_dict = self.network.state_dict()

        for fault in fault_list:
            if f"{fault.layer_name}._packed_params._packed_params" in state_dict:
                packed_params = state_dict[f"{fault.layer_name}._packed_params._packed_params"]
                weight_tensor = packed_params[0].dequantize()
            else:
                weight_tensor = state_dict[f"{fault.layer_name}.weight"]

            print(f"Validating fault for layer '{fault.layer_name}' with index {fault.tensor_index} and weight tensor shape {weight_tensor.shape}.")  

            # Validate tensor indices
            if all(0 <= index < dim for index, dim in zip(fault.tensor_index, weight_tensor.shape)):
                valid_fault_list.append(fault)
            else:
                print(f"Skipping invalid fault with index {fault.tensor_index} for layer '{fault.layer_name}'.")

        return valid_fault_list





