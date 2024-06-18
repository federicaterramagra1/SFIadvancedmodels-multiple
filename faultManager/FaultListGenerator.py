import os
import csv
import math
import numpy as np

from tqdm import tqdm
from ast import literal_eval as make_tuple

from typing import Type

from faultManager.WeightFault import WeightFault
from faultManager.NeuronFault import NeuronFault
from faultManager.modules.InjectableOutputModule import injectable_output_module_class

from torch.nn import Module
import torch
import torchinfo
import SETTINGS
import SETTINGS


class FaultListGenerator:

    def __init__(self,
                 network: Module,
                 network_name: str,
                 device: torch.device,
                 module_class: Type[Module] = None,
                 input_size: torch.Size = None,
                 save_ifm: bool = False):

        self.network = network
        self.network_name = network_name

        self.device = device

        # ---------------------------- momodifiche fate da me (ATTENZIONE) ----------------------------
        # self.feature_maps_layer_names = [name.replace('.weight', '') for name, module in self.network.named_modules()
        #                                  if isinstance(module, module_classes = torch.nn.Conv2d)]
        
        # ---------------------------- ---------------------------- ----------------------------

        # The class of the injectable modules
        # TODO: extend to multiple module class
        self.module_class = module_class
        self.injectable_module_class = injectable_output_module_class(self.module_class)

        # List of injectable modules. Used only for neurons injection
        self.injectable_output_modules_list = None

        # Create the list of injectable module if the module_class is set
        if self.module_class is not None:
            self.__replace_injectable_output_modules(input_size=input_size,
                                                     save_ifm=save_ifm)

        # Name of the injectable layers
        injectable_layer_names = [name.replace('.weight', '') for name, module in self.network.named_modules()
                                  if isinstance(module, self.module_class)]

        # List of the shape of all the layers that contain weight
        self.net_layer_shape = {name.replace('.weight', ''): param.shape for name, param in self.network.named_parameters()
                                if name.replace('.weight', '') in injectable_layer_names}

        # The fault list
        self.fault_list = None

    # @staticmethod
    # def __compute_date_n(N: int,
    #                      p: float = 0.5,
    #                      e: float = 0.01,
    #                      t: float = 2.58):
    #     """
    #     Compute the number of faults to inject according to the DATE09 formula
    #     :param N: The total number of parameters. If None, compute the infinite population version
    #     :param p: Default 0.5. The probability of a fault
    #     :param e: Default 0.01. The desired error rate
    #     :param t: Default 2.58. The desired confidence level
    #     :return: the number of fault to inject
    #     """
    #     if N is None:
    #         return p * (1-p) * t ** 2 / e ** 2
    #     else:
    #         return N / (1 + e ** 2 * (N - 1) / (t ** 2 * p * (1 - p)))




    def __replace_injectable_output_modules(self,
                                            input_size: torch.Size,
                                            save_ifm: bool = False) -> None:
        """
        Replace the target modules with a version that is injectable
        :param input_size: The size of the input of the network. Used to extract the output shape of each layer
        :param save_ifm: Default False. Whether the injectable output module should save the ifm
        """

        modules_to_replace = [(name, module) for name, module in self.network.named_modules() if
                              isinstance(module, self.module_class)]

        # Initialize the list of all the injectable layers
        self.injectable_output_modules_list = list()

        # Create a summary of the network
        summary = torchinfo.summary(self.network,
                                    device=self.device,
                                    input_size=input_size,
                                    verbose=False)

        # Extract the output, input and kernel shape of all the convolutional layers of the network
        output_shapes = [torch.Size(info.output_size) for info in summary.summary_list if
                         isinstance(info.module, self.module_class)]
        input_shapes = [torch.Size(info.input_size) for info in summary.summary_list if
                        isinstance(info.module, self.module_class)]
        kernel_shapes = [info.module.weight.shape for info in summary.summary_list if
                         isinstance(info.module, self.module_class)]

        # Replace all layers with injectable convolutional layers
        for layer_id, (layer_name, layer_module) in enumerate(modules_to_replace):

            layer_module.__class__ = self.injectable_module_class
            layer_module.init_as_copy(device=self.device,
                                      layer_name=layer_name,
                                      input_shape=input_shapes[layer_id],
                                      output_shape=output_shapes[layer_id],
                                      kernel_shape=kernel_shapes[layer_id],
                                      save_ifm=save_ifm)

            # Append the layer to the list
            self.injectable_output_modules_list.append(layer_module)

        


    def update_network(self,
                       network):
        self.network = network
        self.injectable_output_modules_list = [module for module in self.network.modules()
                                               if isinstance(module, self.injectable_module_class)]


    def get_neuron_fault_list(self
                            #   load_fault_list: bool = False,
                            #   save_fault_list: bool = True,
                            #   seed: int = 51195,
                            #   p: float = 0.5,
                            #   e: float = 0.01,
                            #   t: float = 2.58
                            ):
        """
        Generate a fault list for the neurons according to the DATE09 formula
        :param load_fault_list: Default False. Try to load an existing fault list if it exists, otherwise generate it
        :param save_fault_list: Default True. Whether to save the fault list to file
        :param seed: Default 51195. The seed of the fault list
        :param p: Default 0.5. The probability of a fault
        :param e: Default 0.01. The desired error rate
        :param t: Default 2.58. The desired confidence level
        :return: The fault list
        """

        cwd = os.getcwd()

        try:
            
            with open(f'{SETTINGS.FAULT_LIST_PATH}/{SETTINGS.FAULT_LIST_NAME}', newline='') as f_list:
                
                reader = csv.reader(f_list)
                fault_list = list(reader)[1:]
                fault_list = [NeuronFault(layer_name=str(fault[1]),
                                            layer_index=int(fault[2]),
                                            feature_map_index=make_tuple(fault[3]),
                                            value=float(fault[-1])) for fault in fault_list]

            print('Fault list loaded from file')

            # If you don't have to load the fault list raise the Exception and force the generation
          

        except FileNotFoundError:
            print('Fault list not found')
            exit(-1)
            # Initialize the random number generator
            random_generator = np.random.default_rng(seed=seed)

            # Compute how many fault can be injected per layer
            possible_faults_per_layer = [injectable_layer.output_shape[1] * injectable_layer.output_shape[2] * injectable_layer.output_shape[3]
                                         for injectable_layer in self.injectable_output_modules_list]

            # The population of faults
            total_possible_faults = np.sum(possible_faults_per_layer)

            # The percentage of fault to inject in each layer
            probability_per_layer = [possible_faults / total_possible_faults for possible_faults in possible_faults_per_layer]

            # Compute the total number of fault to inject
            n = self.__compute_date_n(N=int(total_possible_faults),
                                      p=p,
                                      t=t,
                                      e=e)

            # Compute the number of fault to inject in each layer
            injected_faults_per_layer = [math.ceil(probability * n) for probability in probability_per_layer]

            fault_list = list()

            pbar = tqdm(zip(injected_faults_per_layer, self.injectable_output_modules_list),
                        desc='Generating fault list',
                        colour='green')

            # For each layer, generate the fault list
            for layer_index, (n_per_layer, injectable_layer) in enumerate(pbar):
                for i in range(n_per_layer):

                    channel = random_generator.integers(injectable_layer.output_shape[1])
                    height = random_generator.integers(injectable_layer.output_shape[2])
                    width = random_generator.integers(injectable_layer.output_shape[3])
                    value = random_generator.random() * 2 - 1

                    fault_list.append(NeuronFault(layer_name=injectable_layer.layer_name,
                                                  layer_index=layer_index,
                                                  feature_map_index=(channel, height, width),
                                                  value=value))

            if save_fault_list:
                os.makedirs(fault_list_filename, exist_ok=True)
                with open(f'{fault_list_filename}/{seed}_neuron_fault_list.csv', 'w', newline='') as f_list:
                    writer_fault = csv.writer(f_list)
                    writer_fault.writerow(['Injection',
                                           'LayerName',
                                           'LayerIndex',
                                           'FeatureMapIndex',
                                           'Value'])
                    for index, fault in enumerate(fault_list):
                        writer_fault.writerow([index, fault.layer_name, fault.layer_index, fault.feature_map_index, fault.value])


            print('Fault List Generated')

        self.fault_list = fault_list
        return fault_list


    def get_weight_fault_list(self
                    ):

        cwd = os.getcwd()

        try:
           
            

            with open(f'{SETTINGS.FAULT_LIST_PATH}/{SETTINGS.FAULT_LIST_NAME}', newline='') as f_list:
                reader = csv.reader(f_list)

                fault_list = list(reader)[1:]

                fault_list = [WeightFault(  injection = int(fault[0]),
                                            layer_name=fault[1],
                                            tensor_index=make_tuple(fault[2]),
                                            bit=int(fault[-1])) for fault in fault_list]

            print('Fault list loaded from file')

            # If you don't have to load the fault list raise the Exception 
       

        except FileNotFoundError:
            
            print(f'Fault list not found: {SETTINGS.FAULT_LIST_PATH}/{SETTINGS.FAULT_LIST_NAME}')
            exit(-1)
  

        self.fault_list = fault_list
        return fault_list