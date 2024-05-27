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
from smartLayers.utils import NoChangeOFMException

from typing import List, Union

from smartLayers.SmartModule import SmartModule
from utils import get_module_by_name



class FaultInjectionManager:

    def __init__(self,
                 network: Module,
                 network_name: str,
                 smart_modules_list: Union[List[SmartModule], None],
                 device: torch.device,
                 loader: DataLoader,
                 clean_output: torch.Tensor,
                 injectable_modules: List[Union[Module, List[Module]]] = None):

        self.network = network
        self.network_name = network_name
        self.loader = loader
        self.device = device

        self.clean_output = clean_output
        self.faulty_output = list()

        # The folder used for the logg
        self.__log_folder = f'log/{self.network_name}/batch_{self.loader.batch_size}'

        # The folder where to save the output
        self.__faulty_output_folder = SETTINGS.FAULTY_OUTPUT_FOLDER

        # The smart modules in the network
        self.__smart_modules_list = smart_modules_list

        # The number of total inferences and the number of skipped inferences
        self.skipped_inferences = 0
        self.total_inferences = 0

        # The weight fault injector
        self.weight_fault_injector = WeightFaultInjector(self.network)

        # The list of injectable module, used only for neuron fault injection
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
        Run a faulty injection campaign for the network. If a layer name is specified, start the computation from that
        layer, loading the input feature maps of the previous layer
        :param fault_model: The faut model for the injection
        :param fault_list: list of fault to inject. One of ['byzantine_neuron', 'stuck-at_params']
        :param first_batch_only: Default False. Debug parameter, if set run the fault injection campaign on the first
        batch only
        :param force_n: Default None. If set, inject only force_n faults in the network
        :param save_output: Default False. Whether to save the output of the network or not
        :param save_ofm: Default False. Whether to save the ofm of the injectable layers
        :param ofm_folder: Default None. The folder where to save the ofms if save_fm is true
        :return: A tuple formed by : (i) a string containing the formatted time elapsed from the beginning to the end of
        the fault injection campaign, (ii) an integer measuring the average memory occupied (in MB)
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
            # This is also important to avoid differences between a
            fault_list = sorted(fault_list, key=lambda x: x.layer_name)

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

                    # Change the description of the progress bar
                    # if fault_dropping and fault_delayed_start:
                    #     pbar.set_description(f'FI (w/ drop & delayed) on b {batch_id}')
                    # elif fault_dropping:
                    #     pbar.set_description(f'FI (w/ drop) on b {batch_id}')
                    # elif fault_delayed_start:
                    #     pbar.set_description(f'FI (w/ delayed) on b {batch_id}')

                    # ----------------------------- #

                    # Inject faults
                    if fault_model == 'byzantine_neuron':
                        injected_layer = self.__inject_fault_on_neuron(fault=fault)
                    elif fault_model == 'stuck-at_params':
                        self.__inject_fault_on_weight(fault, fault_mode='stuck-at')
                    else:
                        raise ValueError(f'Invalid fault model {fault_model}')

                    # Reset memory occupation stats
                    torch.cuda.reset_peak_memory_stats()

                    # If you have to save the ifm, update the file names
                    if save_ofm:
                        for injectable_module in self.injectable_modules:
                            injectable_module.ifm_path = f'{ofm_folder}/fault_{fault_id}_batch_{batch_id}_layer_{injectable_module.layer_name}'

                    # Run inference on the current batch
                    faulty_scores, faulty_indices, different_predictions = self.__run_inference_on_batch(batch_id=batch_id,
                                                                                                         data=data)

                    # Measure the memory occupation
                    memory_occupation = (torch.cuda.max_memory_allocated() + torch.cuda.max_memory_reserved()) // (1024**2)
                    average_memory_occupation = ((total_iterations - 1) * average_memory_occupation + memory_occupation) // total_iterations

                    # If fault prediction is None, the fault had no impact. Use golden predictions
                    if faulty_indices is None:
                        faulty_scores = self.clean_output[batch_id]
                        faulty_indices = batch_clean_prediction_indices

                    # Measure the accuracy of the batch
                    accuracy_batch_dict[fault_id] = float(torch.sum(target.eq(torch.tensor(faulty_indices)))/len(target))

                    # Move the scores to the gpu
                    faulty_scores = faulty_scores.detach().cpu()

                    faulty_prediction_dict[fault_id] = tuple(zip(faulty_indices, faulty_scores))
                    total_different_predictions += different_predictions

                    # Store the faulty prediction if the option is set
                    if save_output:
                        self.faulty_output.append(faulty_scores.numpy())

                    # Measure the loss in accuracy
                    total_predictions += len(batch[0])
                    different_predictions_percentage = 100 * total_different_predictions / total_predictions
                    pbar.set_postfix({'Different': f'{different_predictions_percentage:.6f}%',
                                      'Skipped': f'{100*self.skipped_inferences/self.total_inferences:.2f}%',
                                      'Avg. memory': f'{average_memory_occupation} MB'}
                                     )

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

        except NoChangeOFMException:
            # If the fault doesn't change the output feature map, then simply say that the fault doesn't worsen the
            # network performances for this batch
            faulty_prediction_scores = None
            faulty_prediction_indices = None
            different_predictions = 0
            self.skipped_inferences += 1

        self.total_inferences += 1

        return faulty_prediction_scores, faulty_prediction_indices, different_predictions

    def __inject_fault_on_weight(self,
                                 fault,
                                 fault_mode='stuck-at') -> None:
        """
        Inject a fault in one of the weight of the network
        :param fault: The fault to inject
        :param fault_mode: Default 'stuck-at'. One of either 'stuck-at' or 'bit-flip'. Which kind of fault model to
        employ
        """

        if fault_mode == 'stuck-at':
            self.weight_fault_injector.inject_stuck_at(layer_name=f'{fault.layer_name}.weight',
                                                       tensor_index=fault.tensor_index,
                                                       bit=fault.bit,
                                                       value=fault.value)
        elif fault_mode == 'bit-flip':
            self.weight_fault_injector.inject_bit_flip(layer_name=f'{fault.layer_name}.weight',
                                                       tensor_index=fault.tensor_index,
                                                       bit=fault.bit,)
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