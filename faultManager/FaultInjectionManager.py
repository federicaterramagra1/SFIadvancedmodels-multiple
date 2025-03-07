import os
import csv
import numpy as np
from tqdm import tqdm
from ast import literal_eval as make_tuple
from typing import Type, List
from torch.nn import Module
import SETTINGS
import time
import math
from datetime import timedelta
from torch.utils.data import DataLoader
from faultManager.WeightFaultInjector import WeightFaultInjector
from typing import List, Union
import torch

class FaultInjectionManager:
    def __init__(self,
                 network: Module,
                 network_name: str,
                 device: torch.device,
                 loader: DataLoader,
                 clean_output: torch.Tensor,
                 injectable_modules: List[Union[Module, List[Module]]] = None,
                 num_faults_to_inject: int = 2):
        self.network = network
        self.network_name = network_name
        self.loader = loader
        self.device = device
        self.clean_output = clean_output
        self.faulty_output = list()
        self.num_faults_to_inject = num_faults_to_inject
        print(f"Injecting {self.num_faults_to_inject} faults per batch")  

        self.__log_folder = f'log/{self.network_name}/batch_{self.loader.batch_size}'
        self.__faulty_output_folder = SETTINGS.FAULTY_OUTPUT_FOLDER
        self.skipped_inferences = 0
        self.total_inferences = 0
        self.weight_fault_injector = WeightFaultInjector(self.network)
        self.injectable_modules = injectable_modules

    def run_faulty_campaign_on_weight(self,
                                      fault_model: str,
                                      fault_list: list,
                                      first_batch_only: bool = False,
                                      force_n: int = None,
                                      save_output: bool = False) -> (str, int):
        self.skipped_inferences = 0
        self.total_inferences = 0
        total_iterations = 1

        with torch.no_grad():
            if force_n is not None:
                fault_list = fault_list[:force_n]

            fault_list = sorted(fault_list, key=lambda x: x[0].injection if isinstance(x, list) and len(x) > 0 else float('inf'))

            start_time = time.time()
            accuracy_dict = dict()

            for batch_id, batch in enumerate(self.loader):
                data, target = batch
                data = data.to(self.device)
                accuracy_batch_dict = dict()
                accuracy_dict[batch_id] = accuracy_batch_dict

                pbar = tqdm(range(0, len(fault_list), self.num_faults_to_inject), colour='green', desc=f'FI on b {batch_id}')
                for i in pbar:
                    batch_faults = fault_list[i:i + self.num_faults_to_inject]
                    if fault_model == 'byzantine_neuron':
                        injected_layers = [self.__inject_fault_on_neuron(fault=f) for f in batch_faults]
                    elif fault_model == 'stuck-at_params':
                        self.__inject_fault_on_weight(batch_faults, fault_mode='stuck-at')
                    else:
                        raise ValueError(f'Invalid fault model {fault_model}')

                    faulty_scores, faulty_indices, different_predictions = self.__run_inference_on_batch(batch_id=batch_id, data=data)
                    accuracy_batch_dict[i] = float(torch.sum(target.eq(torch.tensor(faulty_indices))) / len(target))

                    if save_output:
                        self.faulty_output.append(faulty_scores.numpy())

                    # Restore golden values after fault injection
                    for fault in batch_faults:
                        if fault_model == 'byzantine_neuron':
                            for injected_layer in injected_layers:
                                injected_layer.clean_fault()
                        elif fault_model == 'stuck-at_params':
                            self.weight_fault_injector.restore_golden()

                    total_iterations += 1

                if first_batch_only:
                    break

        elapsed = math.ceil(time.time() - start_time)
        return str(timedelta(seconds=elapsed)), total_iterations


    def __inject_fault_on_weight(self, faults, fault_mode='stuck-at'):
        # Flatten the list of faults before passing it to inject_faults()
        flattened_faults = [fault for batch in faults for fault in batch] if isinstance(faults[0], list) else faults
        self.weight_fault_injector.inject_faults(flattened_faults, fault_mode=fault_mode)
    
    def __run_inference_on_batch(self, batch_id: int, data: torch.Tensor):
        try:
            network_output = self.network(data)
            faulty_prediction = torch.topk(network_output, k=1)
            clean_prediction = torch.topk(self.clean_output[batch_id], k=1)

            different_predictions = int(torch.ne(faulty_prediction.indices, clean_prediction.indices).sum())
            faulty_prediction_scores = network_output
            faulty_prediction_indices = [int(fault) for fault in faulty_prediction.indices]

        except RuntimeError as e:
            print(f'FaultInjectionManager: Skipped inference {self.total_inferences} in batch {batch_id}')
            print(e)
            self.skipped_inferences += 1
            return None, None, None

        self.total_inferences += 1
        return faulty_prediction_scores, faulty_prediction_indices, different_predictions
    
    def save_faulty_outputs(self, faulty_tensor_data, batch_id):
        output_file_path = f"{SETTINGS.FAULTY_OUTPUT_FOLDER}/{SETTINGS.FAULT_MODEL}/batch_{batch_id}.npy"
        print(f"Saving faulty output to {output_file_path}")
        np.save(output_file_path, faulty_tensor_data)
