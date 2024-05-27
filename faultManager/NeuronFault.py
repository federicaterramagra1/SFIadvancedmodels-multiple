class NeuronFault:

    def __init__(self,
                 layer_name: str,
                 layer_index: int,
                 feature_map_index: tuple,
                 value: float):

        self.layer_name = layer_name

        self.layer_index = layer_index
        self.feature_map_index = feature_map_index
        self.value = value