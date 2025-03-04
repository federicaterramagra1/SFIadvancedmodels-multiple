import csv
import os
import shutil
import time
import math
from datetime import timedelta
import copy

import SETTINGS
import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from tqdm import tqdm

from faultManager.NeuronFault import NeuronFault
from faultManager.WeightFaultInjector import WeightFaultInjector

from typing import List, Union
import itertools  # Add this import

class FaultInjectionManager:
    def __init__(self,
                 network: Module,
                 network_name: str,
                 device: torch.device,
                 loader: DataLoader,
                 clean_output: torch.Tensor,
                 injectable_modules: List[Union[Module, List[Module]]] = None,
                 num_faults_to_inject: int = 2):  # Number of faults to inject
        self.network = network
        self.network_name = network_name
        self.loader = loader
        self.device = device

        self.clean_output = clean_output
        self.faulty_output = list()

        self.num_faults_to_inject = num_faults_to_inject
        print(f"Number of faults to inject: {self.num_faults_to_inject}")  # Debugging

        self.__log_folder = f'log/{self.network_name}/batch_{self.loader.batch_size}'
        self.__faulty_output_folder = SETTINGS.FAULTY_OUTPUT_FOLDER
        self.skipped_inferences = 0
        self.total_inferences = 0
        self.weight_fault_injector = WeightFaultInjector(self.network)
        self.injectable_modules = injectable_modules

    
    def run_clean_campaign(self):

        pbar = tqdm(self.loader,
                    desc='Clean Inference',
                    colour='green')

        for batch_id, batch in enumerate(pbar):
            data, _ = batch
            data = data.to(self.device)

            self.network(data)


        def run_faulty_campaign_on_weight(self,
                                      fault_model: str,
                                      fault_list: list,
                                      first_batch_only: bool = False,
                                      force_n: int = None,
                                      save_output: bool = False,
                                      save_ofm: bool = False,
                                      ofm_folder: str = None) -> (str, int):
          """
          Run a faulty injection campaign for the network.
          """
          self.skipped_inferences = 0
          self.total_inferences = 0

          total_different_predictions = 0
          total_predictions = 0

          average_memory_occupation = 0
          total_iterations = 1

          with torch.no_grad():
              if force_n is not None:
                  fault_list = fault_list[:force_n]

              # Order the fault list to speed up the injection
              fault_list = sorted(fault_list, key=lambda x: x.injection)

              # Start measuring the time elapsed
              start_time = time.time()

              # The dict measuring the accuracy of each batch
              accuracy_dict = dict()

              # Cycle all the batches in the data loader
              for batch_id, batch in enumerate(self.loader):
                  data, target = batch
                  data = data.to(self.device)

                  # The list of the accuracy of the network for each fault
                  accuracy_batch_dict = dict()
                  accuracy_dict[batch_id] = accuracy_batch_dict

                  faulty_prediction_dict = dict()
                  batch_clean_prediction_scores = [float(fault) for fault in torch.topk(self.clean_output[batch_id], k=1).values]
                  batch_clean_prediction_indices = [int(fault) for fault in torch.topk(self.clean_output[batch_id], k=1).indices]

                  # Inject all the faults in a single batch
                  pbar = tqdm(fault_list,
                              colour='green',
                              desc=f'FI on b {batch_id}',
                              ncols=shutil.get_terminal_size().columns * 2)
                  for fault_id, fault in enumerate(pbar):
                      # Inject faults
                      if fault_model == 'byzantine_neuron':
                          injected_layer = self.__inject_fault_on_neuron(fault=fault)
                      elif fault_model == 'stuck-at_params':
                          self.__inject_fault_on_weight(fault, fault_mode='stuck-at')
                      else:
                          raise ValueError(f'Invalid fault model {fault_model}')

                      # Run inference on the current batch
                      faulty_scores, faulty_indices, different_predictions = self.__run_inference_on_batch(batch_id=batch_id,
                                                                                                          data=data)

                      # Measure the accuracy of the batch
                      accuracy_batch_dict[fault_id] = float(torch.sum(target.eq(torch.tensor(faulty_indices)))/len(target))

                      # Store the faulty prediction if the option is set
                      if save_output:
                          self.faulty_output.append(faulty_scores.numpy())

                      # Clean the fault
                      if fault_model == 'byzantine_neuron':
                          injected_layer.clean_fault()
                      elif fault_model == 'stuck-at_params':
                          self.weight_fault_injector.restore_golden()
                      else:
                          raise ValueError(f'Invalid fault model {fault_model}')

                      # Increment the iteration count
                      total_iterations += 1

                  # Log the accuracy of the batch
                  os.makedirs(f'{self.__log_folder}/{fault_model}', exist_ok=True)
                  log_filename = f'{self.__log_folder}/{fault_model}/batch_{batch_id}.csv'
                  with open(log_filename, 'w') as log_file:
                      log_writer = csv.writer(log_file)
                      log_writer.writerows(accuracy_batch_dict.items())

                  # Save the output to file if the option is set
                  if save_output:
                      os.makedirs(f'{self.__faulty_output_folder}/{fault_model}', exist_ok=True)
                      np.save(f'{self.__faulty_output_folder}/{fault_model}/batch_{batch_id}', self.faulty_output)
                      self.faulty_output = list()

                  # End after only one batch if the option is specified
                  if first_batch_only:
                      break

          # Measure the average accuracy
          average_accuracy_dict = dict()
          for fault_id in range(len(fault_list)):
              fault_accuracy = np.average([accuracy_batch_dict[fault_id] for _, accuracy_batch_dict in accuracy_dict.items()])
              average_accuracy_dict[fault_id] = float(fault_accuracy)

          # Final log
          os.makedirs(f'{self.__log_folder}/{fault_model}', exist_ok=True)
          log_filename = f'{self.__log_folder}/{fault_model}/all_batches.csv'
          with open(log_filename, 'w') as log_file:
              log_writer = csv.writer(log_file)
              log_writer.writerows(average_accuracy_dict.items())

          elapsed = math.ceil(time.time() - start_time)

          return str(timedelta(seconds=elapsed)), average_memory_occupation

    def __run_inference_on_batch(self,
                                 batch_id: int,
                                 data: torch.Tensor):
        try:
            # Execute the network on the batch
            network_output = self.network(data)
            faulty_prediction = torch.topk(network_output, k=1)
            clean_prediction = torch.topk(self.clean_output[batch_id], k=1)

            # Measure the different predictions in terms of scores
            # different_predictions = int(torch.ne(faulty_prediction.values, clean_prediction.values).sum())

            # Measure the different predictions in terms of labels
            different_predictions = int(torch.ne(faulty_prediction.indices, clean_prediction.indices).sum())

            faulty_prediction_scores = network_output
            faulty_prediction_indices = [int(fault) for fault in faulty_prediction.indices]

        except RuntimeError as e:
            print(f'FaultInjectionManager: Skipped inference {self.total_inferences} in batch {batch_id}')
            print(e)
            self.skipped_inferences += 1
            faulty_prediction_scores = None
            faulty_prediction_indices = None
            different_predictions = None

        self.total_inferences += 1

        return faulty_prediction_scores, faulty_prediction_indices, different_predictions

    def __inject_fault_on_weight(self, fault, fault_mode='stuck-at') -> None:
        """
        Inject multiple faults in the weights of the network.
        :param fault: The fault to inject.
        :param fault_mode: The type of fault to inject (e.g., 'stuck-at' or 'bit-flip').
        """
        if fault_mode == 'stuck-at':
            self.weight_fault_injector.inject_faults([fault], fault_mode='stuck-at')
        elif fault_mode == 'bit-flip':
            self.weight_fault_injector.inject_faults([fault], fault_mode='bit-flip')
        else:
            print('FaultInjectionManager: Invalid fault mode')
            quit()


    def __inject_fault_on_neuron(self,
                                 fault: NeuronFault) -> Module:
        """
        Inject a fault in the neuron
        :param fault: The fault to inject
        :return: The injected layer
        """
        output_fault_mask = torch.zeros(size=self.injectable_modules[fault.layer_index].output_shape)

        layer = fault.layer_index
        channel = fault.feature_map_index[0]
        height = fault.feature_map_index[1]
        width = fault.feature_map_index[2]
        value = fault.value

        # Set values to one for the injected elements
        output_fault_mask[0, channel, height, width] = 1

        # Cast mask to int and move to device
        output_fault_mask = output_fault_mask.int().to(self.device)

        # Create a random output
        output_fault = torch.ones(size=self.injectable_modules[layer].output_shape, device=self.device).mul(value)

        # Inject the fault
        self.injectable_modules[layer].inject_fault(output_fault=output_fault,
                                                    output_fault_mask=output_fault_mask)

        # Return the injected layer
        return self.injectable_modules[layer]