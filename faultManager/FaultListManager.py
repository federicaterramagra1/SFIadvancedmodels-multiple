import os
import csv
import numpy as np
from tqdm import tqdm
from ast import literal_eval as make_tuple
from typing import Type, List
from torch.nn import Module
import SETTINGS
from faultManager.WeightFault import WeightFault
from faultManager.NeuronFault import NeuronFault
from faultManager.modules.InjectableOutputModule import injectable_output_module_class
import torch
import os
import csv
import numpy as np
from tqdm import tqdm
from ast import literal_eval as make_tuple
from typing import Type, List
from torch.nn import Module
import SETTINGS
from faultManager.WeightFault import WeightFault
import torch

class FLManager:
    def __init__(self, network: Module, network_name: str, device: torch.device, module_class: Type[Module] = None):
        self.network = network
        self.network_name = network_name
        self.device = device
        self.module_class = module_class  # Store module class
        self.fault_list = None

        # Initialize injectable modules if module_class is provided
        self.injectable_output_modules_list = []
        if self.module_class is not None:
            self.__replace_injectable_output_modules()

    def __replace_injectable_output_modules(self):
        """
        Replaces target modules with injectable versions.
        """
        modules_to_replace = [
            (name, module) for name, module in self.network.named_modules()
            if isinstance(module, self.module_class)
        ]

        self.injectable_output_modules_list = []
        for layer_name, layer_module in modules_to_replace:
            layer_module.__class__ = injectable_output_module_class(self.module_class)
            self.injectable_output_modules_list.append(layer_module)

    def get_weight_fault_list(self) -> List[List[WeightFault]]:
        """
        Get the fault list for the weights, ensuring valid indices and bit positions.
        """
        fault_list = []
        try:
            with open(f'{SETTINGS.FAULT_LIST_PATH}/{SETTINGS.FAULT_LIST_NAME}', newline='') as f_list:
                reader = csv.reader(f_list)
                fault_list = list(reader)[1:]  # Skip header
                fault_list = [
                    WeightFault(
                        injection=int(fault[0]),
                        layer_name=fault[1],
                        tensor_index=make_tuple(fault[2]),  # Make sure this index matches your weights' shape
                        # Handle multiple bits for each fault. If NUM_FAULTS_TO_INJECT is 2, inject 2 bits per weight
                        bits = list(map(int, fault[3].split(',')))[:SETTINGS.NUM_FAULTS_TO_INJECT]  # Take exactly NUM_FAULTS_TO_INJECT bits
                    )
                    for fault in fault_list
                ]
            print(f'Loaded {len(fault_list)} faults from the fault list')

        except FileNotFoundError:
            print(f'Fault list file not found: {SETTINGS.FAULT_LIST_PATH}/{SETTINGS.FAULT_LIST_NAME}')
            exit(-1)

        # Group the faults into batches for multiple fault injections
        grouped_fault_list = [
            fault_list[i:i + SETTINGS.NUM_FAULTS_TO_INJECT]  # Group faults into batches based on the injection size
            for i in range(0, len(fault_list), SETTINGS.NUM_FAULTS_TO_INJECT)
        ]

        return grouped_fault_list

