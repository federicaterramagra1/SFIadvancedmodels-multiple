class WeightFault:
    def __init__(self,
                 injection: int,
                 layer_name: str,
                 tensor_index: tuple,
                 bits: list,  # Ensure bits is a list
                 value: int = None):
        self.injection = injection
        self.layer_name = layer_name
        self.tensor_index = tensor_index
        self.bits = bits if isinstance(bits, list) else [bits]  # Ensure bits is always a list
        self.value = value
        
    def _print(self):
        print(f'Fault on layer {self.layer_name} tensor index {self.tensor_index} bits {self.bits} value {self.value}')
