class WeightFault:

    def __init__(self,
                 injection : int,
                 layer_name: str,
                 tensor_index: tuple,
                 bit: int,
                 value: int = None):
        self.injection = injection
        self.layer_name = layer_name
        self.tensor_index = tensor_index
        self.bit = bit
        self.value = value
        
    def _print(self):
        print(f'Fault on layer {self.layer_name} tensor index {self.tensor_index} bit {self.bit} value {self.value}')