import pickle

import torch
from torch import Tensor
from torch.nn import Module

from smartLayers.utils import NoChangeOFMException


class SmartModule(Module):

    def __init__(self,
                 module: Module,
                 device: torch.device,
                 input_size: torch.Size,
                 output_size: torch.Size,
                 layer_name: str,
                 fm_folder: str,
                 threshold: float = 0) -> None:

        super(SmartModule, self).__init__()

        # The masked convolutional layer
        self.__module = module

        # The device used for inference
        self.__device = device

        # Size of input, output and kernel tensors
        self.output_size = output_size
        self.input_size = input_size

        # The id of the batch currently used for inference
        self.__batch_id = None

        # The name of the layer
        self.layer_name = layer_name

        # The path of the folder where the output and input feature map file (containing the tensors) are located
        self.__fm_folder = fm_folder

        # The golden input/output of the layer
        self.__golden_ifm = None

        # Whether the output of this layer should be compared with the golden output
        self.__compare_ifm_with_golden = False

        # Whether to use the input from the previous layer or the input from the saved ifm
        self.__start_from_this_layer = False

        # The threshold under which a fault ha no impact
        self.__threshold = threshold

    @staticmethod
    def __check_difference(golden: torch.Tensor,
                           faulty: torch.Tensor,
                           threshold: float):
        """
        If faulty contains at least one nan, raise NoChangeOFMException. If no element of the faulty tensor has a distance
        from the same of element of the golden tensor greater than threshold, raise a NoChangeOFMException
        :param golden: The golden tensor
        :param faulty: The faulty tensor
        :param threshold: The threshold
        :return:
        """

        if threshold == 0:
            if torch.all(faulty.eq(golden)):
                raise NoChangeOFMException

        elif torch.sum((golden - faulty).abs() > threshold) == 0:
            raise NoChangeOFMException

    def get_golden_ifm(self):
        return self.__golden_ifm


    def load_golden(self,
                    batch_id: int) -> None:
        """
        Load the golden output feature map from disk, store it into GPU or CPU memory
        :param batch_id: The index of the batch currently used for inference
        """

        self.__batch_id = batch_id

        # Name of the ifm file
        ifm_file_name = f'{self.__fm_folder}/ifm_batch_{self.__batch_id}_layer_{self.layer_name}.pt'

        # Load the golden ifm
        with open(ifm_file_name, 'rb') as ifm_file:
            self.__golden_ifm = pickle.load(ifm_file).to(self.__device)

    def unload_golden(self) -> None:
        """
        Delete all the stored golden ifm
        """
        if self.__golden_ifm is not None:
            del self.__golden_ifm
            self.__golden_ifm = None


    def start_from_this_layer(self) -> None:
        """
        Mark this layer as the initial one for starting the inference, meaning that the input of the layer is not the
        output of the previous layer but the value of the ifm
        """
        self.__start_from_this_layer = True

    def do_not_start_from_this_layer(self) -> None:
        """
        Mark this layer as not the initial one for the inference, meaning that the input of the layer is the output of
        the previous layer
        """
        self.__start_from_this_layer = False

    def compare_with_golden(self) -> None:
        """
        Mark the layer as comparable, so that the faulty output is compared with the golden output at run time
        """
        # Mark the layer as comparable
        self.__compare_ifm_with_golden = True


    def do_not_compare_with_golden(self) -> None:
        """
        Mark the layer as non-comparable
        """
        self.__compare_ifm_with_golden = False

    def forward(self,
                input_tensor: Tensor) -> Tensor:
        """
        Mask of the actual forward function
        :param input_tensor: The input tensor of the layer
        :return: The output tensor of the layer
        """

        if self.__start_from_this_layer:
            input_tensor = self.__golden_ifm
        elif self.__compare_ifm_with_golden:
            # Check for difference with the golden input, if the layer is marked
            self.__check_difference(golden=self.__golden_ifm,
                                    faulty=input_tensor,
                                    threshold=self.__threshold)

        # Compute convolutional output
        output_tensor = self.__module(input_tensor)

        return output_tensor