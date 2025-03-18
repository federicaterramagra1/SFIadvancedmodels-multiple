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
            Get the fault list for the weights, ensuring exactly NUM_FAULTS_TO_INJECT per weight.
            """
            fault_list = []
            try:
                with open(f'{SETTINGS.FAULT_LIST_PATH}/{SETTINGS.FAULT_LIST_NAME}', newline='') as f_list:
                    reader = csv.reader(f_list)
                    fault_list = list(reader)[1:]  # Skip header

                    # Group faults by weight
                    grouped_faults = {}
                    for fault in fault_list:
                        tensor_index = make_tuple(fault[2])
                        layer_name = fault[1]
                        bits = list(map(int, fault[3].split(',')))[:SETTINGS.NUM_FAULTS_TO_INJECT]

                        key = (layer_name, tensor_index)
                        if key not in grouped_faults:
                            grouped_faults[key] = []

                        if len(grouped_faults[key]) < SETTINGS.NUM_FAULTS_TO_INJECT:
                            grouped_faults[key].append(WeightFault(
                                injection=int(fault[0]),
                                layer_name=layer_name,
                                tensor_index=tensor_index,
                                bits=bits
                            ))

                    grouped_fault_list = list(grouped_faults.values())

                print(f'✅ Loaded {len(grouped_fault_list)} unique fault groups from the fault list')

            except FileNotFoundError:
                print(f'❌ ERROR: Fault list file not found: {SETTINGS.FAULT_LIST_PATH}/{SETTINGS.FAULT_LIST_NAME}')
                exit(-1)

            return grouped_fault_list



