import copy
from functools import reduce
import types

from typing import List, Tuple, Type

import torchinfo
import torch
from torch.nn import Module

from faultManager.WeightFault import WeightFault
from smartLayers.SmartModule import SmartModule

from smartLayers.utils import get_delayed_start_module_subclass


class SmartModulesManager:

    def __init__(self,
                 network: Module,
                 device: torch.device,
                 input_size: torch.Size = torch.Size((1, 3, 32, 32)),
                 delayed_start_module: Module = None):

        self.network = network
        self.delayed_start_module = delayed_start_module

        self.device = device

        # Create a summary of the network
        self.summary = torchinfo.summary(self.network,
                                         device=self.device,
                                         input_size=input_size,
                                         verbose=False)


    @staticmethod
    def __generate_layers(self) -> None:
        """
        Generate a list of all the children modules contained in this module
        """
        self.layers = [children for name, children in self.named_children()]


    def replace_module_forward(self) -> None:
        """
        Replace the module forward function with a smart version
        """

        # Add the starting layer attribute
        self.delayed_start_module.starting_layer = None
        self.delayed_start_module.starting_module = None

        # Inherit the delayed start module to a child module that overloads the forward function
        self.delayed_start_module.__class__ = get_delayed_start_module_subclass(superclass_type=type(self.delayed_start_module))

        # If not present, add the costume generate_layer_function, otherwise resort to the module implementation of this
        # function
        if not callable(getattr(self.delayed_start_module, "generate_layer_list", None)):
            self.delayed_start_module.generate_layer_list = types.MethodType(SmartModulesManager.__generate_layers, self.delayed_start_module)

        # Generate the layer list
        self.delayed_start_module.generate_layer_list()


    def replace_smart_modules(self,
                              module_classes: Tuple[Type[Module]] or Type[Module],
                              fm_folder: str,
                              threshold: float = 0,
                              fault_list: List[WeightFault] = None) -> List[SmartModule]:
        """
        Replace all the module_classes layers of the network with injectable module_classes layers
        :param module_classes: The type (or tuple of types) of module to replace
        :param fm_folder: The folder containing the input and output feature maps
        :param threshold: The threshold under which a folder has no impact
        :param fault_list: Default None. If specified, update the name of the layer in the fault list to reflect the
        substitution of layers
        :return A list of all the new InjectableConv2d
        """

        # Select where to look for the module to replace
        if self.delayed_start_module is not None:
            modules_to_replace_parent = self.delayed_start_module
        else:
            modules_to_replace_parent = self.network

        # Find a list of all the layers that need to be replaced by a smart module. A layer needs to be replaced if it
        # is an instance of module_classes and if it is a children of delayed_start_module (or network if it is None).
        # The name must include all the layers: for this reason the layer must be searched among all the network modules
        modules_to_replace = [(name, copy.deepcopy(module)) for name, module in self.network.named_modules()
                              if isinstance(module, module_classes)
                              and module in modules_to_replace_parent.children()]


        # Extract the output, input and kernel shape of all the convolutional layers of the network
        output_sizes = [torch.Size(info.output_size) for info in self.summary.summary_list
                        if isinstance(info.module, module_classes)]
        input_sizes = [torch.Size(info.input_size) for info in self.summary.summary_list
                       if isinstance(info.module, module_classes)]

        # Initialize the list of all the injectable layers
        smart_modules_list = list()

        # Replace all convolution layers with injectable convolutional layers
        for layer_id, (layer_name, layer_module) in enumerate(modules_to_replace):
            # To fine the actual layer with nested layers (e.g. inside a convolutional layer inside a Basic Block in a
            # ResNet, first separate the layer names using the '.'
            formatted_names = layer_name.split(sep='.')

            # If there are more than one names as a result of the separation, the Module containing the convolutional layer
            # is a nester Module
            if len(formatted_names) > 1:
                # In this case, access the nested layer iteratively using itertools.reduce and getattr
                container_layer = reduce(getattr, formatted_names[:-1], self.network)
            else:
                # Otherwise, the containing layer is the network itself (no nested blocks)
                container_layer = self.network


            # Create the injectable version of the convolutional layer
            smart_module = SmartModule(module=layer_module,
                                       device=self.device,
                                       layer_name=layer_name,
                                       input_size=input_sizes[layer_id],
                                       output_size=output_sizes[layer_id],
                                       fm_folder=fm_folder,
                                       threshold=threshold)

            # Append the layer to the list
            smart_modules_list.append(smart_module)

            # Change the convolutional layer to its injectable counterpart
            setattr(container_layer, formatted_names[-1], smart_module)

            # Update the fault list with the new name
            if fault_list is not None:
                smart_module_name = [name for name, module in self.network.named_modules() if module is smart_module][0]
                for fault in fault_list:
                    if '._SmartModule__module' not in fault.layer_name:
                        fault.layer_name = fault.layer_name.replace(smart_module_name, f'{smart_module_name}._SmartModule__module')

        # If the network has a layer list, regenerate to update the layers in the list
        if self.delayed_start_module is not None and callable(getattr(self.delayed_start_module, "generate_layer_list", None)):
            self.delayed_start_module.generate_layer_list()

        return smart_modules_list