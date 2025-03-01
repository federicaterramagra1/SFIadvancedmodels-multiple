import copy
import os
import torch
import SETTINGS
from faultManager.FaultListManager import FLManager
from faultManager.FaultInjectionManager import FaultInjectionManager
from ofmapManager.OutputFeatureMapsManager import OutputFeatureMapsManager
from utils import get_network, get_device, get_loader, get_fault_list, clean_inference, output_definition,  \
                  get_fault_list, clean_inference, output_definition,  fault_list_gen, csv_summary
   


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
        device = get_device(use_cuda0=SETTINGS.USE_CUDA_0,
                            use_cuda1=SETTINGS.USE_CUDA_1)
        
        print(f'Using device {device}')
         
        # Load the dataset
        _, loader = get_loader(network_name=SETTINGS.NETWORK,
                            batch_size=SETTINGS.BATCH_SIZE,
                            dataset_name=SETTINGS.DATASET)
        
        # Load the network
        network = get_network(network_name=SETTINGS.NETWORK,
                            device=device,
                            dataset_name=SETTINGS.DATASET)

        # Apply 8-bit static quantization to the network
        if hasattr(network, 'quantize'):  # Check if the network has a quantize method
            print("Applying 8-bit static quantization to the network...")
            network.quantize()  # Quantize the model
            device = 'cpu'  # Quantized models only support CPU
            network.to(device)  # Move the quantized model to CPU
            print("Quantization completed. Model is now running on CPU.")
        else:
            print("The network does not support quantization. Skipping quantization.")
        
        if SETTINGS.ONLY_CLEAN_INFERENCE:
            print('clean inference accuracy test:')
            clean_inference(network, loader, device, SETTINGS.NETWORK)
            exit(-1)
        
        print('clean inference accuracy test:')
        clean_inference(network, loader, device, SETTINGS.NETWORK)

        # Folder containing the feature maps
        clean_fm_folder = SETTINGS.CLEAN_FM_FOLDER
        faulty_fm_folder = SETTINGS.FAULTY_FM_FOLDER
        
        os.makedirs(clean_fm_folder, exist_ok=True)
        os.makedirs(faulty_fm_folder, exist_ok=True)

        # Folder containing the clean output
        clean_output_folder = SETTINGS.CLEAN_OUTPUT_FOLDER

        #attenzione a module_classes che mi salva ofm diverse!
        module_classes = SETTINGS.MODULE_CLASSES
        
        feature_maps_layer_names = [name.replace('.weight', '') for name, module in network.named_modules()
                                            if isinstance(module, module_classes)]
        
        print('feature maps layer names:')
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

        # Create a smart network. a copy of the network with its convolutional layers replaced by their smart counterpart
        # smart_network = copy.deepcopy(network)
        # fault_list_generator.update_network(network)

        # Manage the fault models
        fault_list, injectable_modules = get_fault_list(fault_model=SETTINGS.FAULT_MODEL,
                                                        fault_list_generator=fault_list_generator)

        # Execute the fault injection campaign with the smart network
        fault_injection_executor = FaultInjectionManager(network=network,
                                                        network_name=SETTINGS.NETWORK,
                                                        device=device,
                                                        loader=loader,
                                                        clean_output=clean_ofm_manager.clean_output,
                                                        injectable_modules=injectable_modules)
        
        fault_injection_executor.run_faulty_campaign_on_weight(fault_model=SETTINGS.FAULT_MODEL,
                                                            fault_list=fault_list,
                                                            first_batch_only=False,
                                                            force_n=SETTINGS.FAULTS_TO_INJECT,
                                                            save_output=SETTINGS.SAVE_FAULTY_OUTPUT,
                                                            save_ofm=SETTINGS.SAVE_FAULTY_OFM,
                                                            ofm_folder=faulty_fm_folder)
        
        
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
        print('Generating csv summary')
        csv_summary()
        print('csv summary generated')
        



if __name__ == '__main__':
    main()
# Dummy change to test commit

# Dummy change to test commit
