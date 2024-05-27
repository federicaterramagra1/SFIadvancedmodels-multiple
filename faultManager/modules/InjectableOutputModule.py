import pickle

import numpy as np
import torch
from torch import Tensor
import SETTINGS

def injectable_output_module_class(parent_module):

    class InjectableOutputModule(parent_module):

        def __init__(self,
                     device: torch.device,
                     layer_name: str,
                     input_shape: torch.Size = None,
                     output_shape: torch.Size = None,
                     kernel_shape: torch.Size = None,
                     save_ifm: bool = False,
                     save_compressed: bool = False
                     ):

            super().__init__()


            # # ---------------------------- momodifiche fate da me (ATTENZIONE) ----------------------------
            # self.feature_maps_layer_names = [name.replace('.weight', '') for name, module in self.network.named_modules()
            #                              if isinstance(module, module_classes = torch.nn.Conv2d)]
            # # # ---------------------------- ---------------------------- ---------------------------- ----------------------------

            # --- PRIVATE PARAMETERS --- #

            # The mask of the clean output
            self.__output_clean_mask = None

            # The value of the fault and the related mask
            self.__output_fault_mask = None
            self.__output_fault = None

            # Whether to save the IFM of a layer
            self.__save_ifm = save_ifm
            self.__save_compressed = save_compressed

            # --- PUBLIC PARAMETERS --- #

            # Name of this layer
            self.layer_name = layer_name

            # Shapes of input, output and kernel
            self.input_shape = input_shape
            self.output_shape = output_shape
            self.kernel_shape = kernel_shape

            # The device where to run the injection
            self.device = device

            # The path where to save the ifm
            self.ifm_path = None


        def init_as_copy(self,
                         device: torch.device,
                         layer_name: str,
                         input_shape: torch.Size = None,
                         output_shape: torch.Size = None,
                         kernel_shape: torch.Size = None,
                         save_ifm: bool = False,
                         save_compressed: bool = False):

            # --- PRIVATE PARAMETERS --- #
            self.__output_clean_mask = None
            self.__output_fault_mask = None
            self.__output_fault = None
            self.__save_ifm = save_ifm
            self.__save_compressed = save_compressed

            # --- PUBLIC PARAMETERS --- #
            self.layer_name = layer_name
            self.input_shape = input_shape
            self.output_shape = output_shape
            self.kernel_shape = kernel_shape
            self.device = device
            self.ifm_path = None

        def inject_fault(self,
                         output_fault: Tensor,
                         output_fault_mask: Tensor = None) -> None:
            """
            Inject a fault in the output feature map of the layer
            :param output_fault: the fault to add to the clean output of the module
            :param output_fault_mask: Default None. If not specified, the mask is automatically created from the
            output_fault: the mask contains zero where the output_fault is zero and one otherwise
            """
            self.__output_fault = output_fault

            # The fault mask can be implicitly computed from the fault output
            if output_fault_mask is None:
                self.__output_fault_mask = self.__output_fault.ne(0).int()
            else:
                self.__output_fault_mask = output_fault_mask

            # The clean_mask is 1 - fault_mask. Basically, it needs to zero all the values affected by a fault in the output
            self.__output_clean_mask = (torch.ones(size=self.__output_fault_mask.shape, device=self.device) - self.__output_fault_mask).int()


        def clean_fault(self):
            """
            Remove an injected fault from the output feature map of the layer
            """

            self.__output_fault_mask = None


        def forward(self,
                    input_tensor: Tensor) -> Tensor:
            """
            The forward function of an injectable module execute the module forward function without applying any additional
            values if the module is not affected by any fault. Otherwise, it sums the clean mask times the clean output +
            the fault output * fault mask:
                output = mask_clean * output_clean + mask_fault * output_fault
            :param input_tensor: The input tensor of the module
            :return: The module forward affected by a fault is the module is affected by a fault
            """
            
            # Save the input to file and self.layer_name == "layer1.0.conv1" (ResNet20) self.feature_maps_layer_names[]
            # MobileNetV2 first layer: layers.0.conv1
            # ResNet20 first layer: layer1.0.conv1
            if self.__save_ifm and self.layer_name == SETTINGS.INPUT_FMAPS_TO_SAVE:
                
                if self.__save_compressed:
                    np.savez_compressed(self.ifm_path, input_tensor.detach().cpu().numpy())
                else:
                    np.savez(self.ifm_path, input_tensor.detach().cpu().numpy())

            # output_clean = self.__clean_module(input_tensor)
            output_clean = super().forward(input_tensor)

            if self.__output_fault_mask is not None:

                masked_output_clean = torch.mul(output_clean, self.__output_clean_mask)
                masked_output_fault = torch.mul(self.__output_fault, self.__output_fault_mask)

                # This operation is needed when you do not have the output normalized between -1 and 1: the effect is
                # that the neuron has a random value inside its range
                max_tensor = torch.max(output_clean)
                min_tensor = torch.min(output_clean)
                range_width = torch.abs(max_tensor - min_tensor)
                masked_output_fault = torch.mul(masked_output_fault, range_width)

                return masked_output_clean + masked_output_fault
            else:
                return output_clean

    return InjectableOutputModule