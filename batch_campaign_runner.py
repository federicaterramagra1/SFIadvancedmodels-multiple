import copy
import os
import torch
import SETTINGS
import random
from faultManager.FaultListManager import FLManager
from faultManager.FaultInjectionManager import FaultInjectionManager
from ofmapManager.OutputFeatureMapsManager import OutputFeatureMapsManager
from utils import (
    get_network, get_device, get_loader, get_fault_list,
    clean_inference, output_definition, fault_list_gen,
    csv_summary, num_experiments_needed, select_random_faults, train_model, faulty_inference
)

def print_layer_dimensions(network):
    for name, param in network.named_parameters():
        if 'weight' in name:
            print(f"Layer {name} weight shape: {param.shape}")

torch.backends.quantized.engine = 'fbgemm'

def main():
    if SETTINGS.FAULT_LIST_GENERATION:
        fault_list_gen()

    if SETTINGS.FAULTS_INJECTION or SETTINGS.ONLY_CLEAN_INFERENCE:
        device = get_device(use_cuda0=SETTINGS.USE_CUDA_0, use_cuda1=SETTINGS.USE_CUDA_1)
        print(f'Using device {device}')

        _, loader = get_loader(network_name=SETTINGS.NETWORK, batch_size=SETTINGS.BATCH_SIZE, dataset_name=SETTINGS.DATASET)

        network = get_network(network_name=SETTINGS.NETWORK, device=device, dataset_name=SETTINGS.DATASET)
        print(f"Network structure:\n{network}")
        print_layer_dimensions(network)

        train_loader, val_loader = get_loader(
            network_name=SETTINGS.NETWORK,
            batch_size=SETTINGS.BATCH_SIZE,
            dataset_name=SETTINGS.DATASET
        )

        print("Valutazione modello prima della quantizzazione:")
        clean_inference(network, loader, device, SETTINGS.NETWORK)

        print(" Inizio training del modello...")
        model_save_path = f"./trained_models/{SETTINGS.DATASET_NAME}_{SETTINGS.NETWORK}_trained.pth"
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        network = train_model(
            model=network,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=SETTINGS.NUM_EPOCHS if hasattr(SETTINGS, 'NUM_EPOCHS') else 15,
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

        _, loader = get_loader(network_name=SETTINGS.NETWORK, batch_size=SETTINGS.BATCH_SIZE, dataset_name=SETTINGS.DATASET)

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

        # BLOCCO: fault injection partizionata
        current_part = SETTINGS.PART_ID
        part_fault_list_name = f"{SETTINGS.NETWORK}_{SETTINGS.SEED}_fault_list_part_{current_part}.csv"
        SETTINGS.FAULT_LIST_NAME = part_fault_list_name

        output_csv_path = os.path.join(SETTINGS.FI_ANALYSIS_PATH, f"output_analysis_part_{current_part}.csv")
        # 🔧 Crea la cartella se non esiste
        os.makedirs(SETTINGS.FI_ANALYSIS_PATH, exist_ok=True)
        if os.path.exists(output_csv_path):
            with open(output_csv_path, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    print(f" Parte {current_part} già completata, salto.")
                    return
                else:
                    print(f"File {output_csv_path} vuoto o incompleto. Rigenero.")
                    os.remove(output_csv_path)


        print(f"\n Avvio iniezione per blocco PART_ID={current_part}/{SETTINGS.NUM_PARTS - 1}")

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
            save_output=True,
            part_id=current_part  # salvataggio batch_{i}_part_{current_part}.npy
        )

        # OUTPUT ANALYSIS
        output_definition(test_loader=loader, batch_size=SETTINGS.BATCH_SIZE, part_id=current_part)
        print(f" Analisi PART_ID={current_part} salvata in {output_csv_path}")

    # ANALISI FINALE
    if SETTINGS.FI_ANALYSIS_SUMMARY:
        print('Generating CSV summary...')
        csv_summary()
        print('CSV summary generated')

if __name__ == '__main__':
    main()
