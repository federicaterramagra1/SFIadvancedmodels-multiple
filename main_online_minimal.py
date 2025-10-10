import os
import time
import random
import numpy as np
import torch
import SETTINGS

from faultManager.FaultListManager import FLManager
from faultManager.FaultInjectionManager import FaultInjectionManager
from ofmapManager.OutputFeatureMapsManager import OutputFeatureMapsManager
from utils import (
    get_network, get_device, get_loader, get_fault_list, load_from_dict,
    clean_inference, output_definition, fault_list_gen,
    csv_summary, save_global_metrics_summary_txt, num_experiments_needed, select_random_faults,
    train_model_complete, faulty_inference, _init_clean_output,
    output_definition_parallel, output_definition_parallel_chunked,
    csv_summary_parallel_chunked, csv_summary_parallel, train_model
)

# ======================= Flag & default coerenti =======================
USE_ADVANCED_TRAIN   = getattr(SETTINGS, "USE_ADVANCED_TRAIN", True)
DO_PTQ               = getattr(SETTINGS, "DO_PTQ", True)            # applicata solo se il modello espone quantize_model()
CALIB_SPLIT          = getattr(SETTINGS, "CALIB_SPLIT", "train")    # "train" o "val"
DO_REFIT             = getattr(SETTINGS, "DO_REFIT", True)
REFIT_EPOCHS         = getattr(SETTINGS, "REFIT_EPOCHS", 30)
EARLY_STOP_PATIENCE  = getattr(SETTINGS, "EARLY_STOP_PATIENCE", 80)
NUM_EPOCHS           = getattr(SETTINGS, "NUM_EPOCHS", 200)
LR_MAIN              = getattr(SETTINGS, "LR_MAIN", 1e-3)

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # niente full determinism per non rallentare troppo
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

def print_layer_dimensions(network):
    for name, param in network.named_parameters():
        if 'weight' in name:
            print(f"Layer {name} weight shape: {tuple(param.shape)}")

def main():
    start_time = time.time()
    set_global_seed(getattr(SETTINGS, "SEED", 123))

    # Motore quantizzazione x86; usato solo se si quantizza davvero
    torch.backends.quantized.engine = 'fbgemm'

    # 1) Generazione fault list (se richiesto)
    if getattr(SETTINGS, "FAULT_LIST_GENERATION", False):
        fault_list_gen()

    # 2) Inference/training/FI
    loader = None
    if SETTINGS.FAULTS_INJECTION or SETTINGS.ONLY_CLEAN_INFERENCE:
        device = get_device(use_cuda0=SETTINGS.USE_CUDA_0, use_cuda1=SETTINGS.USE_CUDA_1)
        print(f'Using device {device}')

        # --- rete + loader unici (split fissi per tutto il run) ---
        network = get_network(network_name=SETTINGS.NETWORK, device=device, dataset_name=SETTINGS.DATASET)
        print(f"Network structure:\n{network}")
        print_layer_dimensions(network)

        train_loader, val_loader, test_loader = get_loader(
            network_name=SETTINGS.NETWORK,
            batch_size=SETTINGS.BATCH_SIZE,
            dataset_name=SETTINGS.DATASET
        )
        loader = test_loader  # useremo SEMPRE il test per clean/FI

        # --- valutazione PRE (float, prima del training) ---
        network.eval()
        print("\n[PRE] Valutazione prima di training/quantizzazione (float, test set):")
        clean_inference(network, loader, device, SETTINGS.NETWORK)

        # --- checkpoint path ---
        model_save_path = f"./trained_models/{SETTINGS.DATASET}_{SETTINGS.NETWORK}_trained.pth"
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        # --- training o load ---
        if os.path.exists(model_save_path):
            print(f"[CKPT] trovato {model_save_path}. Carico pesi FLOAT...")
            load_from_dict(network, device, model_save_path)
        else:
            print("[Train] Avvio training...")
            if USE_ADVANCED_TRAIN:
                network = train_model_complete(
                    model=network,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    num_epochs=NUM_EPOCHS,
                    lr=LR_MAIN,
                    early_stop_patience=EARLY_STOP_PATIENCE,
                    device=device,
                    save_path=model_save_path,
                    do_ptq=False,                 # PTQ uniforme fatta qui sotto se disponibile
                    calib_loader=None,
                    REFIT_TRAINVAL=DO_REFIT,
                    REFIT_EPOCHS=REFIT_EPOCHS
                )
            else:
                network = train_model(
                    model=network,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    num_epochs=NUM_EPOCHS,
                    lr=LR_MAIN,
                    device=device,
                    save_path=model_save_path
                )

        # --- PTQ SOLO se disponibile & richiesto ---
        did_ptq = False
        if DO_PTQ and hasattr(network, 'quantize_model') and callable(network.quantize_model) \
           and not getattr(network, '_quantized_done', False):
            print("\n[PTQ] Applying 8-bit static quantization to the network...")
            calib_loader = train_loader if CALIB_SPLIT == "train" else (val_loader or train_loader)
            network.to('cpu').eval()
            network.quantize_model(calib_loader=calib_loader)
            network._quantized_done = True
            did_ptq = True
            print("[PTQ] Quantization completed. Model is now on CPU.")

        # se non ho quantizzato, rimango sul device originario
        if did_ptq:
            device = torch.device('cpu')
        network.eval()

        # --- valutazione POST (stessi loader, stesso split) ---
        print("\n[POST] Valutazione dopo training/quantizzazione (test set):")
        clean_inference(network, loader, device, SETTINGS.NETWORK)

        # --- Preparazione FI (usa SEMPRE gli stessi loader) ---
        os.makedirs(SETTINGS.CLEAN_FM_FOLDER, exist_ok=True)
        os.makedirs(SETTINGS.FAULTY_FM_FOLDER, exist_ok=True)
        clean_output_folder = SETTINGS.CLEAN_OUTPUT_FOLDER

        clean_ofm_manager = OutputFeatureMapsManager(
            network=network,
            loader=loader,
            module_classes=SETTINGS.MODULE_CLASSES,
            device=device,
            fm_folder=SETTINGS.CLEAN_FM_FOLDER,
            clean_output_folder=clean_output_folder
        )
        clean_ofm_manager.load_clean_output(force_reload=True)

        # --- Fault list coerente ---
        fl_manager = FLManager(
            network=network,
            network_name=SETTINGS.NETWORK,
            device=device,
            module_class=SETTINGS.MODULE_CLASSES_FAULT_LIST
        )
        fault_list, injectable_modules = get_fault_list(
            fault_model=SETTINGS.FAULT_MODEL,
            fault_list_generator=fl_manager
        )

        # opzionale: limitazione riproducibile
        if SETTINGS.FAULTS_TO_INJECT is not None and 0 < SETTINGS.FAULTS_TO_INJECT < len(fault_list):
            random.seed(SETTINGS.SEED)
            fault_list = random.sample(fault_list, SETTINGS.FAULTS_TO_INJECT)
            print(f" Fault list limitata a {SETTINGS.FAULTS_TO_INJECT} fault.")
        else:
            print(f" Fault list ESAUSTIVA: {len(fault_list)} fault groups (ognuno da {SETTINGS.NUM_FAULTS_TO_INJECT} bit).")

        # --- Esecuzione campagna FI ---
        fi_exec = FaultInjectionManager(
            network=network,
            network_name=SETTINGS.NETWORK,
            device=device,
            loader=loader,
            clean_output=clean_ofm_manager.clean_output,
            injectable_modules=injectable_modules,
            num_faults_to_inject=SETTINGS.NUM_FAULTS_TO_INJECT
        )
        fi_exec.run_faulty_campaign_on_weight(
            fault_model=SETTINGS.FAULT_MODEL,
            fault_list=fault_list,
            first_batch_only=False,
            save_output=True
        )

    # 3) Analisi post-campagna
    if getattr(SETTINGS, "FI_ANALYSIS", False):
        try:
            output_definition_parallel(test_loader=loader, batch_size=SETTINGS.BATCH_SIZE, n_workers=8)
            print('Done')
        except Exception:
            print('No loader found to save the labels, creating a new one (same split function).')
            _, _, loader = get_loader(network_name=SETTINGS.NETWORK, batch_size=SETTINGS.BATCH_SIZE, dataset_name=SETTINGS.DATASET)
            _init_clean_output(os.path.join(SETTINGS.CLEAN_OUTPUT_FOLDER, 'clean_output.npy'))
            output_definition_parallel_chunked(test_loader=loader, batch_size=SETTINGS.BATCH_SIZE, n_workers=16)
            print('Done')

    if getattr(SETTINGS, "FI_ANALYSIS_SUMMARY", False):
        print('Generating CSV summary')
        csv_summary()
        print('CSV summary generated')
        save_global_metrics_summary_txt()

    minutes, seconds = divmod(time.time() - start_time, 60)
    print(f"\n Tempo totale di esecuzione: {int(minutes)} minuti e {seconds:.2f} secondi")

if __name__ == '__main__':
    main()
