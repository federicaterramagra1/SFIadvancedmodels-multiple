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
)
from faultManager.WeightFaultInjector import WeightFaultInjector
from tqdm import tqdm
import pandas as pd
from faultManager.WeightFault import WeightFault


def run_online_fault_injection():
    start_time = time.time()

    # FORCE CPU ONLY
    device = torch.device('cpu')
    print(f"Using device: {device}")

    # NETWORK (from utils, already with quant wrapper)
    model = get_network(SETTINGS.NETWORK_NAME, device, SETTINGS.DATASET_NAME)

    # LOAD CHECKPOINT (if any)
    if getattr(SETTINGS, "LOAD_MODEL_FROM_PATH", False):
        load_from_dict(model, device, SETTINGS.LOAD_MODEL_PATH)

    # DATALOADER
    _, _, test_loader = get_loader(
        SETTINGS.NETWORK_NAME,
        SETTINGS.BATCH_SIZE,
        dataset_name=SETTINGS.DATASET_NAME
    )
    print("Test loader ready.")

    # QUANTIZATION
    if hasattr(model, 'quantize_model') and callable(model.quantize_model):
        print("Applying quantization using model.quantize_model...")
        model.quantize_model(calib_loader=test_loader)
        print("Quantization completed. Model is now running on CPU.")
    else:
        print("WARNING: model.quantize_model not found. Skipping quantization.")

    # Compute CLEAN output first
    print("Computing clean predictions...")
    clean_predictions = []
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)
            clean_predictions.append(preds.cpu())
    clean_predictions = torch.cat(clean_predictions)

    # LOAD FAULT LIST
    if not hasattr(SETTINGS, "FAULT_LIST_NAME") or SETTINGS.FAULT_LIST_NAME is None:
        SETTINGS.FAULT_LIST_NAME = f"SimpleMLP_42_fault_list_N{SETTINGS.NUM_FAULTS_TO_INJECT}.csv"

    fault_list_path = os.path.join(SETTINGS.FAULT_LIST_PATH, SETTINGS.FAULT_LIST_NAME)

    if not os.path.exists(fault_list_path):
        print(f"Fault list {fault_list_path} not found. Generating...")
        from utils import fault_list_gen
        fault_list_gen()
        assert os.path.exists(fault_list_path), f"Fault list was not created: {fault_list_path}"
    if not hasattr(SETTINGS, "FAULT_LIST_NAME") or SETTINGS.FAULT_LIST_NAME is None:
        SETTINGS.FAULT_LIST_NAME = f"SimpleMLP_42_fault_list_N{SETTINGS.NUM_FAULTS_TO_INJECT}.csv"
    fault_list_path = os.path.join(SETTINGS.FAULT_LIST_PATH, SETTINGS.FAULT_LIST_NAME)
    fault_df = pd.read_csv(fault_list_path)
    fault_groups = list(fault_df.groupby("Injection"))

    if SETTINGS.FAULTS_TO_INJECT != -1:
        fault_groups = fault_groups[:SETTINGS.FAULTS_TO_INJECT]
        print(f"Subsampling to {SETTINGS.FAULTS_TO_INJECT} faults")

    print(f"Running ONLINE FI campaign with {len(fault_groups)} faults")

    # FAULT INJECTOR
    injector = WeightFaultInjector(model)
    results = []

    for inj_id, group in tqdm(fault_groups, total=len(fault_groups), desc="Online FI"):
        faults = [
            WeightFault(
                injection=inj_id,
                layer_name=row['Layer'],
                tensor_index=eval(row['TensorIndex']),
                bits=[int(row['Bit'])]
            ) for _, row in group.iterrows()
        ]

        masked = 0
        critical = 0
        total = 0
        faulty_predictions = []

        with torch.no_grad():
            injector.inject_faults(faults, fault_mode='bit-flip')

            for data, labels in test_loader:
                data = data.to(device)
                outputs = model(data)
                preds = torch.argmax(outputs, dim=1)
                faulty_predictions.append(preds.cpu())

            injector.restore_golden()

        faulty_predictions = torch.cat(faulty_predictions)

        # Compare clean vs faulty predictions
        total = len(clean_predictions)
        masked = (faulty_predictions == clean_predictions).sum().item()
        critical = (faulty_predictions != clean_predictions).sum().item()

        results.append([inj_id, masked, critical, total])

    # SAVE CSV
    os.makedirs(SETTINGS.FI_ANALYSIS_PATH, exist_ok=True)
    output_path = os.path.join(
        SETTINGS.FI_ANALYSIS_PATH,
        f"online_summary_N{SETTINGS.NUM_FAULTS_TO_INJECT}.csv"
    )
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Injection', 'masked', 'critical', 'total', 'accuracy', 'failure_rate'])
        for inj_id, masked, critical, total in results:
            accuracy = masked / total if total > 0 else 0.0
            failure_rate = critical / total if total > 0 else 0.0
            writer.writerow([inj_id, masked, critical, total, round(accuracy, 4), round(failure_rate, 4)])

    duration = time.time() - start_time
    print(f"\nAnalisi ONLINE completata in {duration/60:.2f} minuti.")


if __name__ == "__main__":
    run_online_fault_injection()
