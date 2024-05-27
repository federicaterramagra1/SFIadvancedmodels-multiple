from typing import Type

import torch
from torch.nn import Linear


class NoChangeOFMException(Exception):
    """
    Exception thrown when the execution of a faulty neural network doesn't produce any difference between its own
    output feature map and the output feature map of a clean network execution
    """
    pass


def get_delayed_start_module_subclass(superclass_type: Type) -> Type:
    """
    Return the class dynamically extended from the module class type
    :param superclass_type: The type of the superclass, used to extend it
    :return:
    """

    # Define a DelayedStartModule class that dynamically extends the delayed_start_module_class to support an
    # overloading of the forward method, while being able to call the parent forward method
    class DelayedStartModule(superclass_type):

        def __init__(self):
            super(DelayedStartModule).__init__()

            self.layers = None

            self.starting_layer = None
            self.starting_module = None

        def forward(self,
                    input_tensor: torch.Tensor) -> torch.Tensor:
            """
            Smart forward used for fault delayed start. With this smart function, the inference starts from the first layer
            marked as starting layer and the input of that layer is loaded from disk
            :param input_tensor: The module input tensor
            :return: The module output tensor
            """

            # If the starting layer and starting module are set, proceed with the smart forward
            if self.starting_layer is not None:
                # Execute the layers iteratively, starting from the one where the fault is injected
                layer_index = self.layers.index(self.starting_layer)
            else:
                layer_index = 0

            if self.starting_module is not None:
                # Create a dummy input
                x = torch.zeros(size=self.starting_module.input_size, device='cuda')

                # Specify that the first module inside this layer should load the input from memory and not read from previous
                # layer
                self.starting_module.start_from_this_layer()
            else:
                x = input_tensor

            # Manage the case nno f.c. -> f.c.
            previous_fc = False

            # Iteratively execute modules in the layer
            for layer in self.layers[layer_index:]:
                # Check whether the current layer is fully connected
                current_fc = isinstance(layer, Linear)

                # If this is a fully connected layer, the previous layer was not and the output is not mono-dimensional,
                # flatten the output
                if not previous_fc and current_fc and len(x.shape) > 1:
                    x = layer(x.flatten(1))
                else:
                    x = layer(x)

                # Update the helper variable
                previous_fc = current_fc

            if self.starting_module is not None:
                # Clear the marking on the first module
                self.starting_module.do_not_start_from_this_layer()

            return x

    return DelayedStartModule