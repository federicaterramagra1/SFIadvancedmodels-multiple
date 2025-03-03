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




class FaultInjectionManager:

    def __init__(self,
                 network: Module,
                 network_name: str,
                 device: torch.device,
                 loader: DataLoader,
                 clean_output: torch.Tensor,
                 injectable_modules: List[Union[Module, List[Module]]] = None,
                 num_faults_to_inject: int = 1):  # Aggiunto il parametro num_faults_to_inject
        self.network = network
        self.network_name = network_name
        self.loader = loader
        self.device = device

        self.clean_output = clean_output
        self.faulty_output = list()

        # Nuovo attributo
        self.num_faults_to_inject = SETTINGS.FAULTS_TO_INJECT  # Memorizza il numero di guasti


        # The folder used for the logg
        self.__log_folder = f'log/{self.network_name}/batch_{self.loader.batch_size}'

        # The folder where to save the output
        self.__faulty_output_folder = SETTINGS.FAULTY_OUTPUT_FOLDER

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
    Run a faulty injection campaign for the network with support for multiple faults injection (1, 2, or 3).
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

        # Ordina la lista di guasti
        fault_list = sorted(fault_list, key=lambda x: x.injection)

        start_time = time.time()

        accuracy_dict = dict()

        # Ciclo sui batch nel data loader
        for batch_id, batch in enumerate(self.loader):
            data, target = batch
            data = data.to(self.device)

            accuracy_batch_dict = dict()
            accuracy_dict[batch_id] = accuracy_batch_dict

            faulty_prediction_dict = dict()
            batch_clean_prediction_scores = [float(fault) for fault in torch.topk(self.clean_output[batch_id], k=1).values]
            batch_clean_prediction_indices = [int(fault) for fault in torch.topk(self.clean_output[batch_id], k=1).indices]

            # Inietta il numero richiesto di guasti in un singolo batch
            pbar = tqdm(fault_list,
                        colour='green',
                        desc=f'FI on b {batch_id}',
                        ncols=shutil.get_terminal_size().columns * 2)
            for fault_id, fault in enumerate(pbar):
                # Logica per iniettare il numero richiesto di guasti (1, 2 o 3)
                for _ in range(self.num_faults_to_inject):
                    # ----------------------------- #
                    # Inietta i guasti in base al modello
                    if fault_model == 'byzantine_neuron':
                        injected_layer = self.__inject_fault_on_neuron(fault=fault)
                    elif fault_model == 'stuck-at_params':
                        self.__inject_fault_on_weight(fault, fault_mode='stuck-at')
                    else:
                        raise ValueError(f'Invalid fault model {fault_model}')

                    # Inizia a misurare l'occupazione della memoria
                    torch.cuda.reset_peak_memory_stats()

                    # Se è necessario salvare gli IFM, aggiorna i nomi dei file
                    if save_ofm:
                        for injectable_module in self.injectable_modules:
                            injectable_module.ifm_path = f'{ofm_folder}/fault_{fault_id}_batch_{batch_id}_layer_{injectable_module.layer_name}'

                    # Esegui l'inferenza con i guasti iniettati
                    faulty_scores, faulty_indices, different_predictions = self.__run_inference_on_batch(batch_id=batch_id,
                                                                                                         data=data)

                    # Misura l'occupazione della memoria
                    memory_occupation = (torch.cuda.max_memory_allocated() + torch.cuda.max_memory_reserved()) // (1024**2)
                    average_memory_occupation = ((total_iterations - 1) * average_memory_occupation + memory_occupation) // total_iterations

                    # Se la previsione con i guasti è None, significa che non c'è stato impatto
                    if faulty_indices is None:
                        faulty_scores = self.clean_output[batch_id]
                        faulty_indices = batch_clean_prediction_indices

                    # Calcola la precisione per batch
                    accuracy_batch_dict[fault_id] = float(torch.sum(target.eq(torch.tensor(faulty_indices)))/len(target))

                    # Memorizza i risultati delle previsioni con i guasti
                    faulty_scores = faulty_scores.detach().cpu()
                    faulty_prediction_dict[fault_id] = tuple(zip(faulty_indices, faulty_scores))
                    total_different_predictions += different_predictions

                    # Salva l'output se richiesto
                    if save_output:
                        self.faulty_output.append(faulty_scores.numpy())

                    # Aggiorna il progresso
                    total_predictions += len(batch[0])
                    different_predictions_percentage = 100 * total_different_predictions / total_predictions
                    pbar.set_postfix({'Different': f'{different_predictions_percentage:.6f}%',
                                      'Skipped': f'{100*self.skipped_inferences/self.total_inferences:.2f}%',
                                      'Avg. memory': f'{average_memory_occupation} MB'}
                                     )

                    # Rimuovi il guasto iniettato
                    if fault_model == 'byzantine_neuron':
                        injected_layer.clean_fault()
                    elif fault_model == 'stuck-at_params':
                        self.weight_fault_injector.restore_golden()
                    else:
                        raise ValueError(f'Invalid fault model {fault_model}')

                    total_iterations += 1

            # Log delle metriche di accuratezza per il batch
            os.makedirs(f'{self.__log_folder}/{fault_model}', exist_ok=True)
            log_filename = f'{self.__log_folder}/{fault_model}/batch_{batch_id}.csv'
            with open(log_filename, 'w') as log_file:
                log_writer = csv.writer(log_file)
                log_writer.writerows(accuracy_batch_dict.items())

            # Salva l'output dei guasti se richiesto
            if save_output:
                os.makedirs(f'{self.__faulty_output_folder}/{fault_model}', exist_ok=True)
                np.save(f'{self.__faulty_output_folder}/{fault_model}/batch_{batch_id}', self.faulty_output)
                self.faulty_output = list()

            # Se necessario, termina dopo il primo batch
            if first_batch_only:
                break

    # Calcola la media dell'accuratezza
    average_accuracy_dict = dict()
    for fault_id in range(len(fault_list)):
        fault_accuracy = np.average([accuracy_batch_dict[fault_id] for _, accuracy_batch_dict in accuracy_dict.items()])
        average_accuracy_dict[fault_id] = float(fault_accuracy)

    # Log finale
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
