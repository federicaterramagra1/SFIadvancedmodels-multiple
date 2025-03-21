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
from faultManager.WeightFault import WeightFault 
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

    def run_faulty_campaign_on_weight(self, fault_model: str, fault_list: list, first_batch_only: bool = False, save_output: bool = True):
        """
        Esegue la fault injection a batch, applicando per ogni gruppo le iniezioni in sequenza.
        """
        print(f"DEBUG: Running fault injection campaign with {len(fault_list)} fault groups")
        
        with torch.no_grad():
            for batch_id, batch in enumerate(self.loader):
                data, _ = batch
                data = data.to(self.device)
                faulty_outputs_batch = []
                pbar = tqdm(fault_list, colour='green', desc=f'FI on batch {batch_id}')
                
                for fault_group in tqdm(fault_list, desc=f'FI on batch {batch_id}'):
                    print(f"DEBUG: Processing fault group for {fault_group[0].layer_name}, index {fault_group[0].tensor_index}")
                    # applica tutte le iniezioni cumulative
                    self.weight_fault_injector.inject_faults(fault_group, fault_mode=fault_model)
                    # inferenza UNA SOLA volta
                    faulty_scores, _, _ = self.__run_inference_on_batch(batch_id, data)
                    faulty_outputs_batch.append(faulty_scores.cpu().numpy())
                    # restore golden dopo il gruppo
                    self.weight_fault_injector.restore_golden()

                
                if save_output:
                    faulty_outputs_batch = np.array(faulty_outputs_batch) 
                    self.save_faulty_outputs(faulty_outputs_batch, batch_id)
                
                if first_batch_only:
                    break




    def __inject_fault_on_weight(self, faults, fault_mode='stuck-at'):
        """
        Inject a fault into the weights of the network.
        """
        print(f"DEBUG: Received faults - {faults}")  # Print for debugging

        if not isinstance(faults, list):
            print(f"❌ ERROR: Expected a list, but got {type(faults)} - {faults}")
            return  # Avoid crashing the program

        if all(isinstance(fault, WeightFault) for fault in faults):
            flattened_faults = faults  # Already a correct list
        else:
            print(f"❌ ERROR: Found non-WeightFault elements in faults: {faults}")
            return

        print(f"DEBUG: Injecting faults: {[f.layer_name for f in flattened_faults]}")  # Print fault details
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

