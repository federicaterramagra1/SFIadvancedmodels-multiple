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
import os
import numpy as np
import torch

def ensure_directory_exists(directory):
    """
    Ensure that the directory exists. If not, create it.
    """
    if not os.path.exists(directory):
        print(f"Creating missing directory: {directory}")
        os.makedirs(directory)


class FaultInjectionManager:
    def __init__(self,
                 network: torch.nn.Module,
                 network_name: str,
                 device: torch.device,
                 loader: torch.utils.data.DataLoader,
                 clean_output: torch.Tensor,
                 injectable_modules: List[Union[torch.nn.Module, List[torch.nn.Module]]] = None,
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
                                      save_output: bool = False):

        self.skipped_inferences = 0
        self.total_inferences = 0

        internal_fault_mode = 'stuck-at' if fault_model == 'stuck-at_params' else 'flip'

        with torch.no_grad():
            for batch_id, batch in enumerate(self.loader):
                data, target = batch
                data = data.to(self.device)

                faulty_outputs_batch = []

                pbar = tqdm(range(0, len(fault_list), self.num_faults_to_inject), colour='green', desc=f'FI on batch {batch_id}')
                for i in pbar:
                    batch_faults = fault_list[i:i + self.num_faults_to_inject]
                    self.__inject_fault_on_weight(batch_faults, fault_mode=internal_fault_mode)

                    faulty_scores, faulty_indices, _ = self.__run_inference_on_batch(batch_id=batch_id, data=data)

                    # Accumula output difettosi per ogni fault
                    faulty_outputs_batch.append(faulty_scores.cpu().numpy())

                    self.weight_fault_injector.restore_golden()

                # Salva una volta alla fine di ogni batch
                if save_output:
                    faulty_outputs_batch = np.array(faulty_outputs_batch)  # (n_faults, batch_size, outputs)
                    self.save_faulty_outputs(faulty_outputs_batch, batch_id)


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
    
    def save_faulty_outputs(self, faulty_output_data, batch_id):
        batch_folder = f"{SETTINGS.FAULTY_OUTPUT_FOLDER}/{SETTINGS.FAULT_MODEL}"
        os.makedirs(batch_folder, exist_ok=True)
        output_file_path = f"{batch_folder}/batch_{batch_id}.npy"
        np.save(output_file_path, faulty_output_data)
        print(f"Saved faulty output to {output_file_path}")

