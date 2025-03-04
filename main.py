import copy
import os
import torch
import SETTINGS
from faultManager.FaultListManager import FLManager
from faultManager.FaultInjectionManager import FaultInjectionManager
from ofmapManager.OutputFeatureMapsManager import OutputFeatureMapsManager
from utils import get_network, get_device, get_loader, get_fault_list, clean_inference, output_definition, fault_list_gen, csv_summary


def main():
    torch.backends.quantized.engine = 'qnnpack'  # Use 'fbgemm' if you're on an x86 CPU

    if SETTINGS.FAULT_LIST_GENERATION:
        fault_list_gen()
    else:
        print('Fault list generation is disabled')
    
    if SETTINGS.FAULTS_INJECTION or SETTINGS.ONLY_CLEAN_INFERENCE:
        # Set deterministic algorithms
        torch.use_deterministic_algorithms(mode=True)

        # Select the device
        device = get_device(use_cuda0=SETTINGS.USE_CUDA_0, use_cuda1=SETTINGS.USE_CUDA_1)
        print(f'Using device {device}')
        
        # Load the dataset
        _, loader = get_loader(network_name=SETTINGS.NETWORK,
                            batch_size=SETTINGS.BATCH_SIZE,
                            dataset_name=SETTINGS.DATASET)
        
        # Load the network
        network = get_network(network_name=SETTINGS.NETWORK,
                            device=device,
                            dataset_name=SETTINGS.DATASET)

        # Debugging: Print the network and check if it supports quantization
        #print(network)
        print(f"Does the network support quantization? {hasattr(network, 'quantize') and callable(network.quantize)}")

        # Apply quantization if supported
        if hasattr(network, 'quantize') and callable(network.quantize):
            print("Applying 8-bit static quantization to the network...")
            # Move the model to the CPU before quantization
            device = 'cpu'
            network.to(device)
            network.quantize()  # Quantize the model
            print("Quantization completed. Model is now running on CPU.")
        else:
            print("The network does not support quantization. Skipping quantization.")
            
        if SETTINGS.ONLY_CLEAN_INFERENCE:
            print('Clean inference accuracy test:')
            clean_inference(network, loader, device, SETTINGS.NETWORK)
            exit(-1)
        
        print('Clean inference accuracy test:')
        clean_inference(network, loader, device, SETTINGS.NETWORK)

        # Folder containing the feature maps
        clean_fm_folder = SETTINGS.CLEAN_FM_FOLDER
        faulty_fm_folder = SETTINGS.FAULTY_FM_FOLDER
        
        os.makedirs(clean_fm_folder, exist_ok=True)
        os.makedirs(faulty_fm_folder, exist_ok=True)

        # Folder containing the clean output
        clean_output_folder = SETTINGS.CLEAN_OUTPUT_FOLDER

        # Feature maps layer names
        module_classes = SETTINGS.MODULE_CLASSES
        feature_maps_layer_names = [name.replace('.weight', '') for name, module in network.named_modules()
                                            if isinstance(module, module_classes)]
        
        print('Feature maps layer names:')
        print(feature_maps_layer_names)
    
        clean_ofm_manager = OutputFeatureMapsManager(network=network,
                                                    loader=loader,
                                                    module_classes=SETTINGS.MODULE_CLASSES,
                                                    device=device,
                                                    fm_folder=clean_fm_folder,
                                                    clean_output_folder=clean_output_folder)

        # Try to load the clean input
        clean_ofm_manager.load_clean_output()

        # Generate fault list
        fault_list_generator = FLManager(network=network,
                                                network_name=SETTINGS.NETWORK,
                                                device=device,
                                                module_class=SETTINGS.MODULE_CLASSES_FAULT_LIST,
                                                input_size=loader.dataset[0][0].unsqueeze(0).shape,
                                                save_ifm=True)

        # Manage the fault models
        fault_list, injectable_modules = get_fault_list(fault_model=SETTINGS.FAULT_MODEL,
                                                        fault_list_generator=fault_list_generator)

        # Execute the fault injection campaign with the smart network
        fault_injection_executor = FaultInjectionManager(
                                                            network=network,
                                                            network_name=SETTINGS.NETWORK,
                                                            device=device,
                                                            loader=loader,
                                                            clean_output=clean_ofm_manager.clean_output,
                                                            injectable_modules=injectable_modules,
                                                            num_faults_to_inject=SETTINGS.NUM_FAULTS_TO_INJECT  # Pass the number of faults to inject
                                                        )
        fault_injection_executor.run_faulty_campaign_on_weight(
                                                            fault_model='stuck-at_params',
                                                            fault_list=fault_list,
                                                            first_batch_only=False,
                                                            force_n=None,
                                                            save_output=True,
                                                            save_ofm=False,
                                                            ofm_folder=None
                                                        )
        
    else:
        print('Fault injection is disabled')
        
    if SETTINGS.FI_ANALYSIS:
        try:
            output_definition(test_loader=loader, batch_size=SETTINGS.BATCH_SIZE)
            print('Done')
        except:
            print('No loader found to save the labels, creating a new one')
            _, loader = get_loader(network_name=SETTINGS.NETWORK,
                            batch_size=SETTINGS.BATCH_SIZE,
                            dataset_name=SETTINGS.DATASET)
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
