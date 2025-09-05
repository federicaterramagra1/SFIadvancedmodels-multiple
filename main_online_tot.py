import torch
import SETTINGS
import time
import os
import csv
from utils import (
    get_loader,
    get_network,
    get_device,
    load_from_dict,
    train_model, 
)
from faultManager.WeightFaultInjector import WeightFaultInjector
from tqdm import tqdm
from faultManager.WeightFault import WeightFault
import random 
import numpy as np

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def run_full_weight_bitflip():
    start_time = time.time()
    device = torch.device('cpu')
    print(f"Using device: {device}")

    # Carica modello già quantizzabile
    model = get_network(SETTINGS.NETWORK_NAME, device, SETTINGS.DATASET_NAME)

    # Path dove salvare o caricare i pesi
    model_save_path = f"./trained_models/{SETTINGS.DATASET_NAME}_{SETTINGS.NETWORK_NAME}_trained.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    if os.path.exists(model_save_path):
        print(f"Modello già addestrato trovato in {model_save_path}. Caricamento in corso...")
        load_from_dict(model, device, model_save_path)
    else:
        print("Inizio training...")
        train_loader, val_loader, test_loader = get_loader(
            network_name=SETTINGS.NETWORK_NAME,
            batch_size=SETTINGS.BATCH_SIZE,
            dataset_name=SETTINGS.DATASET_NAME
        )
        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=getattr(SETTINGS, 'NUM_EPOCHS', 30),
            lr=0.001,
            device=device,
            save_path=model_save_path
        )

    # Loader per il test
    _, _, test_loader = get_loader(
        network_name=SETTINGS.NETWORK_NAME,
        batch_size=SETTINGS.BATCH_SIZE,
        dataset_name=SETTINGS.DATASET_NAME
    )

    # Quantizzazione
    if hasattr(model, 'quantize_model') and callable(model.quantize_model):
        print("Applying quantization...")
        model.quantize_model(calib_loader=test_loader)
        print("Quantization completed.")
    else:
        print("WARNING: model.quantize_model not found. Skipping quantization.")

    # Inferenza pulita
    print("Computing clean predictions...")
    clean_predictions = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)
            clean_predictions.append(preds.cpu())
    clean_predictions = torch.cat(clean_predictions)

    # Fault list FULL FLIP
    print("Generating fault list: flip ALL bits of ALL weights")
    faults = []
    for name, module in model.named_modules():
        if hasattr(module, '_packed_params'):
            try:
                weight, _ = module._packed_params._weight_bias()
                shape = weight.shape
                for idx in torch.cartesian_prod(*[torch.arange(s) for s in shape]):
                    idx_tuple = tuple(idx.tolist())
                    faults.append(WeightFault(
                        injection=0,
                        layer_name=name,
                        tensor_index=idx_tuple,
                        bits=list(range(8))
                    ))
            except Exception as e:
                print(f"Skipping module {name} due to error: {e}")
    print(f"Total weights to flip: {len(faults)}")

    # Iniezione
    injector = WeightFaultInjector(model)
    results = []

    with torch.no_grad():
        first_fault = faults[0]
        layer_name = first_fault.layer_name
        idx = first_fault.tensor_index
        module = dict(model.named_modules())[layer_name]
        weight_before = module.weight()
        print(f"[DEBUG] Peso prima del flip: {weight_before[idx].dequantize().item()}")

        injector.inject_faults(faults, fault_mode=SETTINGS.FAULT_MODEL)

        weight_after = module.weight()
        print(f"[DEBUG] Peso dopo  il flip: {weight_after[idx].dequantize().item()}")

        faulty_predictions = []
        for data, _ in test_loader:
            data = data.to(device)
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)
            faulty_predictions.append(preds.cpu())

        injector.restore_golden()

    faulty_predictions = torch.cat(faulty_predictions)
    unique_classes, counts = torch.unique(faulty_predictions, return_counts=True)
    print(f"Distribuzione classi predette dopo FULL FLIP:")
    for cls, count in zip(unique_classes.tolist(), counts.tolist()):
        print(f"Classe {cls}: {count} predizioni")

    total = len(clean_predictions)
    masked = (faulty_predictions == clean_predictions).sum().item()
    critical = total - masked
    results.append([0, masked, critical, total])

    os.makedirs(SETTINGS.FI_ANALYSIS_PATH, exist_ok=True)
    output_path = os.path.join(
        SETTINGS.FI_ANALYSIS_PATH,
        "online_summary_FULL_FLIP.csv"
    )
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Injection', 'masked', 'critical', 'total', 'accuracy', 'failure_rate'])
        writer.writerow([0, masked, critical, total, round(masked / total, 4), round(critical / total, 4)])

    print(f"\nAnalisi FULL FLIP completata in {round((time.time() - start_time)/60, 2)} minuti.\nRisultato salvato in: {output_path}")

if __name__ == "__main__":
    run_full_weight_bitflip()
