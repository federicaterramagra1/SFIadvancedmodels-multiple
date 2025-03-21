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
        fault_dict = {}
        csv_path = f'{SETTINGS.FAULT_LIST_PATH}/{SETTINGS.FAULT_LIST_NAME}'
        with open(csv_path, newline='') as f:
            reader = csv.reader(f); next(reader)
            for row in reader:
                inj = int(row[0]); layer = row[1]; idx = make_tuple(row[2])
                bits = list(map(int, row[3].split(',')))[:SETTINGS.NUM_FAULTS_TO_INJECT]
                wf = WeightFault(inj, layer, idx, bits)
                fault_dict.setdefault((layer, idx), []).append(wf)
        fault_groups = list(fault_dict.values())
        print(f'✅ Loaded {len(fault_groups)} fault groups (each group={SETTINGS.NUM_FAULTS_TO_INJECT} faults)')
        return fault_groups








