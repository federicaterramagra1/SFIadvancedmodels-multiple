import os
import time
import random
import torch

import SETTINGS

from faultManager.FaultListManager import FLManager
from faultManager.FaultInjectionManager import FaultInjectionManager
from ofmapManager.OutputFeatureMapsManager import OutputFeatureMapsManager

from utils import (
    get_network, get_device, get_loader, get_fault_list, load_from_dict,
    clean_inference, fault_list_gen,
    csv_summary, save_global_metrics_summary_txt,
    train_model_complete, train_model,
    _init_clean_output, output_definition_parallel, output_definition_parallel_chunked,
    csv_summary_parallel, csv_summary_parallel_chunked
)

torch.backends.quantized.engine = "fbgemm"


def print_layer_dimensions(network):
    for name, param in network.named_parameters():
        if "weight" in name:
            print(f"Layer {name} weight shape: {param.shape}")


def main():
    t0 = time.time()

    # -------------------- Fault list (opzionale) --------------------
    if getattr(SETTINGS, "FAULT_LIST_GENERATION", False):
        fault_list_gen()

    run_clean_or_fi = getattr(SETTINGS, "FAULTS_INJECTION", False) or getattr(SETTINGS, "ONLY_CLEAN_INFERENCE", False)
    if run_clean_or_fi:
        # -------------------- Device & rete --------------------
        device = get_device(use_cuda0=SETTINGS.USE_CUDA_0, use_cuda1=SETTINGS.USE_CUDA_1)
        print(f"Using device {device}")

        network = get_network(
            network_name=SETTINGS.NETWORK,
            device=device,
            dataset_name=SETTINGS.DATASET
        )
        print(f"Network structure:\n{network}")
        print_layer_dimensions(network)

        # -------------------- Loader (train/val/test) --------------------
        train_loader, val_loader, test_loader = get_loader(
            network_name=SETTINGS.NETWORK,
            batch_size=SETTINGS.BATCH_SIZE,
            dataset_name=SETTINGS.DATASET
        )
        test_loader_for_eval = test_loader  # usiamo sempre il test per le valutazioni

        # -------------------- Valutazione pre-training --------------------
        print("Valutazione PRIMA del training/quantizzazione (test set):")
        clean_inference(network, test_loader_for_eval, device, SETTINGS.NETWORK)

        # -------------------- Training / Load pesi --------------------
        model_save_path = f"./trained_models/{SETTINGS.DATASET}_{SETTINGS.NETWORK}_trained.pth"
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        if os.path.exists(model_save_path):
            print(f"[CKPT] Trovato {model_save_path}. Carico pesi FLOAT...")
            load_from_dict(network, device, model_save_path)
        else:
            print("[Train] Avvio training...")
            if getattr(SETTINGS, "USE_ADVANCED_TRAIN", False):
                # Training "completo" ma SENZA PTQ qui (uniformiamo la PTQ fuori)
                network = train_model_complete(
                    model=network,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    num_epochs=getattr(SETTINGS, "NUM_EPOCHS", 150),
                    lr=getattr(SETTINGS, "LR_MAIN", 1e-3),
                    early_stop_patience=getattr(SETTINGS, "EARLY_STOP_PATIENCE", 20),
                    device=device,
                    save_path=model_save_path,
                    do_ptq=False,             # <- PTQ uniforme sotto
                    calib_loader=None,
                    USE_CLASS_WEIGHTS=getattr(SETTINGS, "USE_CLASS_WEIGHTS", True),
                    REFIT_TRAINVAL=getattr(SETTINGS, "DO_REFIT", False),
                    REFIT_EPOCHS=getattr(SETTINGS, "REFIT_EPOCHS", 0),
                )
            else:
                network = train_model(
                    model=network,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    num_epochs=getattr(SETTINGS, "NUM_EPOCHS", 50),
                    lr=getattr(SETTINGS, "LR_MAIN", 1e-3),
                    device=device,
                    save_path=model_save_path
                )

        # -------------------- PTQ uniforme (se abilitata) --------------------
        if getattr(SETTINGS, "DO_PTQ", True) and hasattr(network, "quantize_model") and not getattr(network, "_quantized_done", False):
            print("[PTQ] Applying 8-bit static quantization to the network...")
            calib_split = getattr(SETTINGS, "CALIB_SPLIT", "train")
            calib_loader = train_loader if calib_split == "train" else (val_loader or train_loader)

            # PTQ su CPU
            try:
                network.to("cpu").eval()
                maybe_new = network.quantize_model(calib_loader=calib_loader)
                if maybe_new is not None:
                    network = maybe_new
                setattr(network, "_quantized_done", True)
                print("[PTQ] Quantization completed. Model is now on CPU.")
                device = "cpu"
            except Exception as e:
                print(f"[PTQ] Quantizzazione fallita: {e}")
        else:
            # se non PTQ, restiamo sul device originale
            network.to(device)

        # -------------------- Valutazione post-training/quantizzazione --------------------
        print("Valutazione DOPO training/quantizzazione (test set):")
        clean_inference(network, test_loader_for_eval, device, SETTINGS.NETWORK)

        # -------------------- Preparazione FI (feature maps clean) --------------------
        os.makedirs(SETTINGS.CLEAN_FM_FOLDER, exist_ok=True)
        os.makedirs(SETTINGS.FAULTY_FM_FOLDER, exist_ok=True)

        clean_ofm_manager = OutputFeatureMapsManager(
            network=network,
            loader=test_loader_for_eval,
            module_classes=SETTINGS.MODULE_CLASSES,
            device=device,
            fm_folder=SETTINGS.CLEAN_FM_FOLDER,
            clean_output_folder=SETTINGS.CLEAN_OUTPUT_FOLDER
        )
        clean_ofm_manager.load_clean_output(force_reload=True)

        # -------------------- Fault list --------------------
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

        # (opzionale) limita #fault per prova
        if SETTINGS.FAULTS_TO_INJECT is not None and 0 < SETTINGS.FAULTS_TO_INJECT < len(fault_list):
            random.seed(SETTINGS.SEED)
            fault_list = random.sample(fault_list, SETTINGS.FAULTS_TO_INJECT)
            print(f" Fault list limitata a {SETTINGS.FAULTS_TO_INJECT} fault.")
        else:
            print(f" Fault list ESAUSTIVA: {len(fault_list)} fault groups (ognuno da {SETTINGS.NUM_FAULTS_TO_INJECT} bit).")

        # -------------------- Campagna FI --------------------
        if getattr(SETTINGS, "FAULTS_INJECTION", False):
            fi_exec = FaultInjectionManager(
                network=network,
                network_name=SETTINGS.NETWORK,
                device=device,
                loader=test_loader_for_eval,
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

    # -------------------- Analisi post-campagna (se abilitata) --------------------
    if getattr(SETTINGS, "FI_ANALYSIS", False):
        try:
            output_definition_parallel(test_loader=test_loader_for_eval, batch_size=SETTINGS.BATCH_SIZE, n_workers=8)
            print("Done")
        except Exception:
            print("No loader found to save the labels, creating a new one (same split function).")
            _, _, test_loader_for_eval = get_loader(
                network_name=SETTINGS.NETWORK,
                batch_size=SETTINGS.BATCH_SIZE,
                dataset_name=SETTINGS.DATASET
            )
            _init_clean_output(os.path.join(SETTINGS.CLEAN_OUTPUT_FOLDER, "clean_output.npy"))
            output_definition_parallel_chunked(test_loader=test_loader_for_eval, batch_size=SETTINGS.BATCH_SIZE, n_workers=16)
            print("Done")

    if getattr(SETTINGS, "FI_ANALYSIS_SUMMARY", False):
        print("Generating CSV summary")
        # scegli qui la versione che preferisci
        csv_summary()  # semplice
        # csv_summary_parallel(n_workers=8)                 # oppure
        # csv_summary_parallel_chunked(n_workers=8)         # per fault list giganti
        print("CSV summary generated")
        save_global_metrics_summary_txt()

    minutes, seconds = divmod(time.time() - t0, 60)
    print(f"\n Tempo totale di esecuzione: {int(minutes)} minuti e {seconds:.2f} secondi")


if __name__ == "__main__":
    main()
