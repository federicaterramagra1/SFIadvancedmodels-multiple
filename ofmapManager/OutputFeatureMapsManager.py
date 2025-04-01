import os
import shutil

import torch
from torch.nn.modules import Module
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np

from typing import Tuple, Type
import pickle
import SETTINGS


class OutputFeatureMapsManager:

    def __init__(self,
                 network: Module,
                 loader: DataLoader,
                 module_classes: Tuple[Type[Module]] or Type[Module],
                 device: torch.device,
                 fm_folder: str,
                 clean_output_folder: str,
                 save_compressed: bool = False):
        """
        Manges the recording of output feature maps for a given network on a given database
        :param network: The network to analyze
        :param loader: the data loader for which to save the output feature maps
        :param module_classes: The class (or tuple of classes) of the module for which to save the feature maps
        :param device: The device where to perform the inference
        :param clean_output_folder: The folder where to load/store the clean output of the network
        :param fm_folder: The folder containing the input and output feature maps
        :param save_compressed: Default False. Whether to save the ofm compressed
        """
        
        self.network = network
        self.loader = loader
        self.device = device

        # An integer that measures the size of a single batch in memory
        batch = next(iter(self.loader))[0]
        self.batch_memory_occupation = batch.nelement() * batch.element_size()

        # An integer that measures the size weights in memory
        parameters_memory_occupation_list = [parameter.nelement() * parameter.element_size()
                                             for name, parameter in self.network.named_parameters()]
        self.parameters_memory_occupation = np.sum(parameters_memory_occupation_list)

        # A list of all the possible layers. This list is equivalent to all the keys from __output_feature_maps
        self.feature_maps_layer_names = [name.replace('.weight', '') for name, module in self.network.named_modules()
                                         if isinstance(module, module_classes)]
        

        self.feature_maps_layers = [module for name, module in self.network.named_modules()
                                    if name.replace('.weight', '') in self.feature_maps_layer_names]
        
        
        
        # A list of dictionary where every element is the file containing the output feature map for a batch and for the
        # layer
        
        self.__fm_folder = fm_folder
        os.makedirs(self.__fm_folder, exist_ok=True)
        self.ifm_paths = [{j: f'./{fm_folder}/batch_{i}_layer_{j}' for j in self.feature_maps_layer_names} for i in range(0, len(loader))]
        self.__save_compressed = save_compressed

        
     
        # An integer indicating the number of bytes occupied by the Output Feature Maps (without taking into account
        # the overhead required by the lists and the dictionary)
        self.__output_feature_maps_size = 0
        # The equivalent value for input feature maps
        self.__input_feature_maps_size = 0

        # An integer that measures the size of one single batch of Output Feature Maps (without taking into account the
        # overhead required by the dict
        self.output_feature_maps_size = 0
        # The equivalent value for input feature maps
        self.input_feature_maps_size = 0

        # List containing all the registered forward hooks
        self.hooks = list()

        # Tensor containing all the output of all the batches
        self.clean_output = None

        # Name of the file where to save the clean output
        self.__clean_output_folder = clean_output_folder
        os.makedirs(self.__clean_output_folder, exist_ok=True)
        self.__clean_output_path = f'{clean_output_folder}/clean_output.npy'

    def __get_layer_hook(self,
                         batch_id: int,
                         layer_name: str,
                         save_to_cpu: bool):
        """
        Returns a hook function that saves the output feature map of the layer name
        :param batch_id: The index of the current batch
        :param layer_name: Name of the layer for which to save the output feature maps
        :param save_to_cpu: Default True. Whether to save the output feature maps to cpu or not
        :return: the hook function to register as a forward hook
        """
        def save_output_feature_map_hook(_, in_tensor, out_tensor):
            # Move input and output feature maps to main memory and detach
            input_to_save = in_tensor[0].detach().cpu() if save_to_cpu else in_tensor[0].detach()
            output_to_save = out_tensor.detach().cpu() if save_to_cpu else out_tensor.detach()

            # Save the input feature map
            # Dequantize if necessary
            if input_to_save.is_quantized:
                input_to_save = input_to_save.dequantize()

            # Save the input feature map
            if self.__save_compressed:
                np.savez_compressed(self.ifm_paths[batch_id][layer_name], input_to_save.numpy())
            else:
                np.savez(self.ifm_paths[batch_id][layer_name], input_to_save.numpy())


            # Update information about the memory occupation
            self.__input_feature_maps_size += input_to_save.nelement() * input_to_save.element_size()
            self.__output_feature_maps_size += output_to_save.nelement() * output_to_save.element_size()

        return save_output_feature_map_hook

    def __remove_all_hooks(self) -> None:
        """
        Remove all the forward hooks on the network
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = list()

    def save_intermediate_layer_outputs(self,
                                        save_to_cpu: bool = True) -> None:
        """
        Save the intermediate layer outputs of the network for the given dataset, saving the resulting output as a
        dictionary, where each entry corresponds to a layer and contains a list of 4-dimensional tensor NCHW
        :param save_to_cpu: Default True. Whether to save the output feature maps to cpu or not
        """
        
        self.network.eval()
        self.network.to(self.device)

        pbar = tqdm(self.loader, colour='green', desc='Saving Feature Maps')

        clean_output_batch_list = list()
        
        with torch.no_grad():
            for batch_id, batch in enumerate(pbar):
                data, _ = batch
                data = data.to(self.device)
                
                # Register hooks for current batch
                for name, module in self.network.named_modules():
                    
                    if name in self.feature_maps_layer_names and SETTINGS.SAVE_CLEAN_OFM:
                        
                        self.hooks.append(module.register_forward_hook(self.__get_layer_hook(batch_id=batch_id,
                                                                                             layer_name=name,
                                                                                             save_to_cpu=save_to_cpu)))
                # Execute the network and save the clean output
                clean_output_batch = self.network(data).detach().cpu()
                
                clean_output_batch_list.append(clean_output_batch.numpy())

                # Remove all the hooks
                self.__remove_all_hooks()

        # Save the clean output to file
        np.save(self.__clean_output_path, np.array(clean_output_batch_list, dtype=object), allow_pickle=True)
        self.clean_output = [torch.tensor(tensor.astype(np.float32), device=self.device)
                             for tensor in np.load(self.__clean_output_path, allow_pickle=True)]

        self.input_feature_maps_size = self.__input_feature_maps_size / len(self.loader)
        self.output_feature_maps_size = self.__output_feature_maps_size / len(self.loader)

    def load_clean_output(self,
                          force_reload: bool = False) -> None:
        """
        Load the clean output of the network. If the file is not found, compute the clean output (and the clean output
        feature maps)
        """

        if SETTINGS.SAVE_CLEAN_OFM == False:
            try:
                self.clean_output = [torch.tensor(tensor.astype(np.float32), device=self.device)
                                    for tensor in np.load(self.__clean_output_path, allow_pickle=True)]
                
            except FileNotFoundError:
                print('No previous clean output found, starting clean inference...')
                self.save_intermediate_layer_outputs()
        else:
            
            print('No previous clean output or clean feature maps found, starting clean inference...')
            self.save_intermediate_layer_outputs()