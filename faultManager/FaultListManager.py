import os
import csv
import math
import numpy as np

from tqdm import tqdm
from ast import literal_eval as make_tuple

from typing import Type

from faultManager.WeightFault import WeightFault
from faultManager.NeuronFault import NeuronFault
from faultManager.modules.InjectableOutputModule import injectable_output_module_class

from torch.nn import Module
import torch
import torchinfo
import SETTINGS
import SETTINGS


class FLManager:

    def __init__(self,
                 network: Module,
                 network_name: str,
                 device: torch.device,
                 module_class: Type[Module] = None,
                 input_size: torch.Size = None,
                 save_ifm: bool = False):

        self.network = network
        self.network_name = network_name

        self.device = device

        # The class of the injectable modules
        # TODO: extend to multiple module class
        self.module_class = module_class
        self.injectable_module_class = injectable_output_module_class(self.module_class)

        # List of injectable modules. Used only for neurons injection
        self.injectable_output_modules_list = None

        # Create the list of injectable module if the module_class is set
        if self.module_class is not None:
            self.__replace_injectable_output_modules(input_size=input_size,
                                                     save_ifm=save_ifm)

        # Name of the injectable layers
        injectable_layer_names = [name.replace('.weight', '') for name, module in self.network.named_modules()
                                  if isinstance(module, self.module_class)]

        # List of the shape of all the layers that contain weight
        self.net_layer_shape = {name.replace('.weight', ''): param.shape for name, param in self.network.named_parameters()
                                if name.replace('.weight', '') in injectable_layer_names}

        # The fault list
        self.fault_list = None




    def __replace_injectable_output_modules(self,
                                            input_size: torch.Size,
                                            save_ifm: bool = False) -> None:
        """
        Replace the target modules with a version that is injectable
        :param input_size: The size of the input of the network. Used to extract the output shape of each layer
        :param save_ifm: Default False. Whether the injectable output module should save the ifm
        """

        modules_to_replace = [(name, module) for name, module in self.network.named_modules() if
                              isinstance(module, self.module_class)]

        # Initialize the list of all the injectable layers
        self.injectable_output_modules_list = list()

        # Create a summary of the network
        summary = torchinfo.summary(self.network,
                                    device=self.device,
                                    input_size=input_size,
                                    verbose=False)

        # Extract the output, input and kernel shape of all the convolutional layers of the network
        output_shapes = [torch.Size(info.output_size) for info in summary.summary_list if
                         isinstance(info.module, self.module_class)]
        input_shapes = [torch.Size(info.input_size) for info in summary.summary_list if
                        isinstance(info.module, self.module_class)]
        kernel_shapes = [info.module.weight.shape for info in summary.summary_list if
                         isinstance(info.module, self.module_class)]

        # Replace all layers with injectable convolutional layers
        for layer_id, (layer_name, layer_module) in enumerate(modules_to_replace):

            layer_module.__class__ = self.injectable_module_class
            layer_module.init_as_copy(device=self.device,
                                      layer_name=layer_name,
                                      input_shape=input_shapes[layer_id],
                                      output_shape=output_shapes[layer_id],
                                      kernel_shape=kernel_shapes[layer_id],
                                      save_ifm=save_ifm)

            # Append the layer to the list
            self.injectable_output_modules_list.append(layer_module)

        


    def update_network(self,
                       network):
        self.network = network
        self.injectable_output_modules_list = [module for module in self.network.modules()
                                               if isinstance(module, self.injectable_module_class)]


    def get_neuron_fault_list(self
                           
                            ):
        """
        Generate a fault list for the neurons according to the DATE09 formula
        :param load_fault_list: Default False. Try to load an existing fault list if it exists, otherwise generate it
        :param save_fault_list: Default True. Whether to save the fault list to file
        :param seed: Default 51195. The seed of the fault list
        :param p: Default 0.5. The probability of a fault
        :param e: Default 0.01. The desired error rate
        :param t: Default 2.58. The desired confidence level
        :return: The fault list
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

            # If you don't have to load the fault list raise the Exception and force the generation
          

        except FileNotFoundError:
            print('Fault list not found')
            exit(-1)
           


    def get_weight_fault_list(self
                    ):

        cwd = os.getcwd()

        try:
           
            

            with open(f'{SETTINGS.FAULT_LIST_PATH}/{SETTINGS.FAULT_LIST_NAME}', newline='') as f_list:
                reader = csv.reader(f_list)

                fault_list = list(reader)[1:]

                fault_list = [WeightFault(  injection = int(fault[0]),
                                            layer_name=fault[1],
                                            tensor_index=make_tuple(fault[2]),
                                            bit=int(fault[-1])) for fault in fault_list]

            print('Fault list loaded from file')

            # If you don't have to load the fault list raise the Exception 
       

        except FileNotFoundError:
            
            print(f'Fault list not found: {SETTINGS.FAULT_LIST_PATH}/{SETTINGS.FAULT_LIST_NAME}')
            exit(-1)
  

        self.fault_list = fault_list
        return fault_list