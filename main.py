import copy
import os
import torch
import SETTINGS
from faultManager.FaultListManager import FLManager
from faultManager.FaultInjectionManager import FaultInjectionManager
from ofmapManager.OutputFeatureMapsManager import OutputFeatureMapsManager
from utils import (
    get_network, get_device, get_loader, get_fault_list, 
    clean_inference, output_definition, fault_list_gen, 
    csv_summary, num_experiments_needed, select_random_faults
)
# Function to print layer weight dimensions for validation
def print_layer_dimensions(network):
    for name, param in network.named_parameters():
        if 'weight' in name:
            print(f"Layer {name} weight shape: {param.shape}")

import torch
torch.backends.quantized.engine = 'fbgemm'  # Use 'fbgemm' if on x86 CPU

def main():

    if SETTINGS.FAULT_LIST_GENERATION:
        fault_list_gen()
    else:
        print('Fault list generation is disabled')
    
    if SETTINGS.FAULTS_INJECTION or SETTINGS.ONLY_CLEAN_INFERENCE:
        torch.use_deterministic_algorithms(True, warn_only=True)

        device = get_device(use_cuda0=SETTINGS.USE_CUDA_0, use_cuda1=SETTINGS.USE_CUDA_1)
        print(f'Using device {device}')
        
        _, loader = get_loader(network_name=SETTINGS.NETWORK, batch_size=SETTINGS.BATCH_SIZE, dataset_name=SETTINGS.DATASET)
        
        network = get_network(network_name=SETTINGS.NETWORK, device=device, dataset_name=SETTINGS.DATASET)

        print(f"Network structure:\n{network}")
        print_layer_dimensions(network)

        if hasattr(network, 'quantize_model') and callable(network.quantize_model):
            print("Applying 8-bit static quantization to the network...")
            device = 'cpu'
            network.to(device)
            network.quantize_model()
            print("Quantization completed. Model is now running on CPU.")
            
            print("Available layers in state_dict():", network.state_dict().keys())

        else:
            print("The network does not support quantization. Skipping quantization.")
            
        if SETTINGS.ONLY_CLEAN_INFERENCE:
            print('Clean inference accuracy test:')
            clean_inference(network, loader, device, SETTINGS.NETWORK)
            exit(-1)
        
        print('Clean inference accuracy test:')
        clean_inference(network, loader, device, SETTINGS.NETWORK)

        print("Quantized model state dict keys:", network.state_dict().keys())

        clean_fm_folder = SETTINGS.CLEAN_FM_FOLDER
        faulty_fm_folder = SETTINGS.FAULTY_FM_FOLDER
        
        os.makedirs(clean_fm_folder, exist_ok=True)
        os.makedirs(faulty_fm_folder, exist_ok=True)

        clean_output_folder = SETTINGS.CLEAN_OUTPUT_FOLDER

        module_classes = SETTINGS.MODULE_CLASSES
        feature_maps_layer_names = [name.replace('.weight', '') for name, module in network.named_modules() if isinstance(module, module_classes)]
        
        print('Feature maps layer names:')
        print(feature_maps_layer_names)
    
        clean_ofm_manager = OutputFeatureMapsManager(network=network,
                                                     loader=loader,
                                                     module_classes=SETTINGS.MODULE_CLASSES,
                                                     device=device,
                                                     fm_folder=clean_fm_folder,
                                                     clean_output_folder=clean_output_folder)

        clean_ofm_manager.load_clean_output()

        fault_list_generator = FLManager(
            network=network,
            network_name=SETTINGS.NETWORK,
            device=device,
            module_class=SETTINGS.MODULE_CLASSES_FAULT_LIST
        )

        fault_list, injectable_modules = get_fault_list(
            fault_model=SETTINGS.FAULT_MODEL,
            fault_list_generator=fault_list_generator
        )

        print(f"🔍 Fault list generata: {len(fault_list)} faults trovati.")


        # Retrieve injectable modules after initialization
        injectable_modules = fault_list_generator.injectable_output_modules_list

        fault_injection_executor = FaultInjectionManager(network=network,
                                                        network_name=SETTINGS.NETWORK,
                                                        device=device,
                                                        loader=loader,
                                                        clean_output=clean_ofm_manager.clean_output,
                                                        injectable_modules=injectable_modules,
                                                        num_faults_to_inject=SETTINGS.NUM_FAULTS_TO_INJECT)

        fault_injection_executor.run_faulty_campaign_on_weight(
            fault_model=SETTINGS.FAULT_MODEL,
            fault_list=fault_list,
            first_batch_only=False,
            save_output=True
        )

    else:
        print('Fault injection is disabled')
        
    if SETTINGS.FI_ANALYSIS:
        try:
            output_definition(test_loader=loader, batch_size=SETTINGS.BATCH_SIZE)
            print('Done')
        except:
            print('No loader found to save the labels, creating a new one')
            _, loader = get_loader(network_name=SETTINGS.NETWORK, batch_size=SETTINGS.BATCH_SIZE, dataset_name=SETTINGS.DATASET)
            output_definition(test_loader=loader, batch_size=SETTINGS.BATCH_SIZE)
            print('Done')
    else:
        print('Fault injection analysis is disabled')
    
    if SETTINGS.FI_ANALYSIS_SUMMARY:
        print('Generating CSV summary')
        csv_summary()
        print('CSV summary generated')


if __name__ == '__main__':
    main()