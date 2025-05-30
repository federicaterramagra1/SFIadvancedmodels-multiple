import copy
import os
import torch
import SETTINGS
import random
from faultManager.FaultListManager import FLManager
from faultManager.FaultInjectionManager import FaultInjectionManager
from ofmapManager.OutputFeatureMapsManager import OutputFeatureMapsManager
from utils import (
    get_network, get_device, get_loader, get_fault_list, load_from_dict,
    clean_inference, output_definition, fault_list_gen,
    csv_summary, num_experiments_needed, select_random_faults, train_model, faulty_inference, _init_clean_output, output_definition_parallel,output_definition_parallel_chunked , csv_summary_parallel_chunked, csv_summary_parallel
)
import time


def print_layer_dimensions(network):
    for name, param in network.named_parameters():
        if 'weight' in name:
            print(f"Layer {name} weight shape: {param.shape}")

torch.backends.quantized.engine = 'fbgemm'

def main():
    start_time = time.time()

    if SETTINGS.FAULT_LIST_GENERATION:
        fault_list_gen()

    if SETTINGS.FAULTS_INJECTION or SETTINGS.ONLY_CLEAN_INFERENCE:
        device = get_device(use_cuda0=SETTINGS.USE_CUDA_0, use_cuda1=SETTINGS.USE_CUDA_1)
        print(f'Using device {device}')

        # Loader per test set
        _, _, loader = get_loader(network_name=SETTINGS.NETWORK, batch_size=SETTINGS.BATCH_SIZE, dataset_name=SETTINGS.DATASET)

        # Costruisci rete e stampa info
        network = get_network(network_name=SETTINGS.NETWORK, device=device, dataset_name=SETTINGS.DATASET)
        print(f"Network structure:\n{network}")
        print_layer_dimensions(network)

        # Train loaders
        train_loader, val_loader, test_loader = get_loader(
            network_name=SETTINGS.NETWORK,
            batch_size=SETTINGS.BATCH_SIZE,
            dataset_name=SETTINGS.DATASET
        )

        print("Valutazione modello prima della quantizzazione:")
        clean_inference(network, loader, device, SETTINGS.NETWORK)

        print(" Inizio training del modello...")
        model_save_path = f"./trained_models/{SETTINGS.DATASET_NAME}_{SETTINGS.NETWORK}_trained.pth"
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        if os.path.exists(model_save_path):
            print(f"Modello già addestrato trovato in {model_save_path}. Caricamento in corso...")
            load_from_dict(network, device, model_save_path)
        else:
            network = train_model(
                model=network,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=SETTINGS.NUM_EPOCHS if hasattr(SETTINGS, 'NUM_EPOCHS') else 30,
                lr=0.001,
                device=device,
                save_path=model_save_path
            )


        if hasattr(network, 'quantize_model') and callable(network.quantize_model):
            print(" Applying 8-bit static quantization to the network...")
            device = 'cpu'
            network.to(device)
            network.quantize_model(calib_loader=train_loader)
            print(" Quantization completed. Model is now running on CPU.")
        else:
            print("The network does not support quantization. Skipping quantization.")

        if SETTINGS.ONLY_CLEAN_INFERENCE:
            clean_inference(network, loader, device, SETTINGS.NETWORK)
            return

        # RICREA IL LOADER dopo il training e quantizzazione
        _, _, loader = get_loader(network_name=SETTINGS.NETWORK, batch_size=SETTINGS.BATCH_SIZE, dataset_name=SETTINGS.DATASET)

        print("Clean inference accuracy test:")
        clean_inference(network, loader, device, SETTINGS.NETWORK)

        clean_fm_folder = SETTINGS.CLEAN_FM_FOLDER
        faulty_fm_folder = SETTINGS.FAULTY_FM_FOLDER
        os.makedirs(clean_fm_folder, exist_ok=True)
        os.makedirs(faulty_fm_folder, exist_ok=True)

        clean_output_folder = SETTINGS.CLEAN_OUTPUT_FOLDER

        clean_ofm_manager = OutputFeatureMapsManager(
            network=network,
            loader=loader,
            module_classes=SETTINGS.MODULE_CLASSES,
            device=device,
            fm_folder=clean_fm_folder,
            clean_output_folder=clean_output_folder
        )

        clean_ofm_manager.load_clean_output(force_reload=True)

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

        if SETTINGS.FAULTS_TO_INJECT is not None and 0 < SETTINGS.FAULTS_TO_INJECT < len(fault_list):
            fault_list = random.sample(fault_list, SETTINGS.FAULTS_TO_INJECT)
            print(f" Fault list limitata a {SETTINGS.FAULTS_TO_INJECT} fault.")
        else:
            print(f" Fault list ESAUSTIVA: {len(fault_list)} fault groups (ognuno da {SETTINGS.NUM_FAULTS_TO_INJECT} bit).")

        fault_injection_executor = FaultInjectionManager(
            network=network,
            network_name=SETTINGS.NETWORK,
            device=device,
            loader=loader,
            clean_output=clean_ofm_manager.clean_output,
            injectable_modules=injectable_modules,
            num_faults_to_inject=SETTINGS.NUM_FAULTS_TO_INJECT
        )

        fault_injection_executor.run_faulty_campaign_on_weight(
            fault_model=SETTINGS.FAULT_MODEL,
            fault_list=fault_list,
            first_batch_only=False,
            save_output=True
        )


    if SETTINGS.FI_ANALYSIS:
        try:
            output_definition_parallel(test_loader=loader, batch_size=SETTINGS.BATCH_SIZE, n_workers=32)
            print('Done')
        except:
            print('No loader found to save the labels, creating a new one')
            _, _, loader = get_loader(network_name=SETTINGS.NETWORK, batch_size=SETTINGS.BATCH_SIZE, dataset_name=SETTINGS.DATASET)
            
            # inizializza la variabile globale CLEAN_OUTPUT
            _init_clean_output(os.path.join(SETTINGS.CLEAN_OUTPUT_FOLDER, 'clean_output.npy'))

            output_definition_parallel_chunked(test_loader=loader, batch_size=SETTINGS.BATCH_SIZE, n_workers=16)
            print('Done')


    if SETTINGS.FI_ANALYSIS_SUMMARY:
        print('Generating CSV summary')
        csv_summary_parallel_chunked(n_workers=32)
        print('CSV summary generated')
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"\n Tempo totale di esecuzione: {int(minutes)} minuti e {seconds:.2f} secondi")


if __name__ == '__main__':
    main()
