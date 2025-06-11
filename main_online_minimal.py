import torch
import time
from itertools import product, combinations, islice
from math import comb
from tqdm import tqdm
from faultManager.WeightFault import WeightFault
from faultManager.WeightFaultInjector import WeightFaultInjector
from utils import get_loader
import heapq
import SETTINGS
from torch.quantization import QuantWrapper, get_default_qconfig
from dlModels.Banknote.mlp import SimpleMLP

def run_online_fault_injection_minimal():
    start_time = time.time()
    device = torch.device("cpu")
    print(f"Using device: {device}")

    print("Loading raw model and applying quantization manually...")
    base_model = SimpleMLP()
    base_model.qconfig = get_default_qconfig("fbgemm")
    model = QuantWrapper(base_model)
    model.quantize_model = base_model.quantize_model
    model.to(device)

    _, _, test_loader = get_loader("SimpleMLP", SETTINGS.BATCH_SIZE, dataset_name="Banknote")
    print("Test loader ready.")

    if hasattr(model, 'quantize_model') and callable(model.quantize_model):
        print("Applying quantization using model.quantize_model...")
        model.quantize_model(calib_loader=test_loader)
        print("Quantization completed.")
    else:
        print("WARNING: model.quantize_model not found. Skipping quantization.")

    print("Computing clean predictions...")
    clean_predictions = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)
            clean_predictions.append(preds.cpu())
    clean_predictions = torch.cat(clean_predictions)

    print("Generating fault list as in main_online.py...")
    all_faults = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.quantized.Linear):
            weight, _ = module._packed_params._weight_bias()
            shape = weight.shape
            for idx in product(*[range(s) for s in shape]):
                for bit in range(8):
                    all_faults.append((name, idx, bit))

    num_faults = SETTINGS.NUM_FAULTS_TO_INJECT
    total_combinations = comb(len(all_faults), num_faults)
    fault_combinations = combinations(all_faults, num_faults)
    print(f"Total single faults available: {len(all_faults)}")
    print(f"Number of {num_faults}-fault combinations: {total_combinations}")

    batch_size = SETTINGS.BATCH_SIZE
    top_faults = []
    total_failure_rate = 0.0

    # Imposta i punti percentuali di salvataggio ogni 10%
    PERC_SAVE_INTERVAL = 10
    save_points = set([int(total_combinations * x / 100) for x in range(PERC_SAVE_INTERVAL, 101, PERC_SAVE_INTERVAL)])
    output_path = f"top100_faults_progress_N{num_faults}.txt"

    pbar = tqdm(total=total_combinations, desc="Sequential FI")
    for inj_id, combo in enumerate(fault_combinations):
        faults = [WeightFault(injection=inj_id, layer_name=layer, tensor_index=idx, bits=[bit]) for (layer, idx, bit) in combo]
        with torch.no_grad():
            injector = WeightFaultInjector(model)
            injector.inject_faults(faults, fault_mode='bit-flip')
            faulty_predictions = []
            for data, _ in test_loader:
                data = data.to(device)
                outputs = model(data)
                preds = torch.argmax(outputs, dim=1)
                faulty_predictions.append(preds.cpu())
            injector.restore_golden()
        faulty_predictions = torch.cat(faulty_predictions)
        total = len(clean_predictions)
        masked = (faulty_predictions == clean_predictions).sum().item()
        critical = total - masked
        accuracy = masked / total
        failure_rate = critical / total
        total_failure_rate += failure_rate
        if len(top_faults) < 100:
            heapq.heappush(top_faults, (failure_rate, inj_id, faults, accuracy))
        else:
            heapq.heappushpop(top_faults, (failure_rate, inj_id, faults, accuracy))

        # === Salva ogni 10% in un unico file TXT ===
        if inj_id in save_points:
            perc = int(100 * inj_id / total_combinations)
            with open(output_path, 'a') as f:
                f.write(f"\n---- Progress: {perc}% ({inj_id} / {total_combinations}) ----\n")
                f.write(f"Top 100 most critical fault injections (NUM_FAULTS_TO_INJECT = {num_faults})\n")
                f.write(f"Partial average failure rate: {total_failure_rate / (inj_id+1):.4f}\n\n")
                for rank, (failure_rate, inj_id_, faults, accuracy) in enumerate(sorted(top_faults, reverse=True), start=1):
                    fault_desc = ", ".join([f"{fault.layer_name}[{fault.tensor_index}] bit {fault.bits[0]}" for fault in faults])
                    f.write(f"#{rank:3d} | Injection {inj_id_:6d} | FR={failure_rate:.4f} | Acc={accuracy:.4f} | {fault_desc}\n")
            print(f"\n[INFO] Salvato aggiornamento top100 al {perc}% in {output_path}")

        pbar.update(1)
    pbar.close()

    # Salva anche alla fine (100%)
    with open(output_path, 'a') as f:
        f.write(f"\n---- Progress: 100% ({inj_id+1} / {total_combinations}) ----\n")
        f.write(f"Top 100 most critical fault injections (NUM_FAULTS_TO_INJECT = {num_faults})\n")
        f.write(f"Final average failure rate: {total_failure_rate / (inj_id+1):.4f}\n\n")
        for rank, (failure_rate, inj_id_, faults, accuracy) in enumerate(sorted(top_faults, reverse=True), start=1):
            fault_desc = ", ".join([f"{fault.layer_name}[{fault.tensor_index}] bit {fault.bits[0]}" for fault in faults])
            f.write(f"#{rank:3d} | Injection {inj_id_:6d} | FR={failure_rate:.4f} | Acc={accuracy:.4f} | {fault_desc}\n")

    print(f"\nSaved top 100 critical faults (with progress) to {output_path}")

    output_path = f"top100_faults_N{SETTINGS.NUM_FAULTS_TO_INJECT}.txt"
    with open(output_path, 'w') as f:
        f.write(f"Top 100 most critical fault injections (NUM_FAULTS_TO_INJECT = {SETTINGS.NUM_FAULTS_TO_INJECT})\n")
        f.write(f"Average failure rate: {total_failure_rate / (inj_id+1):.4f}\n\n")
        for rank, (failure_rate, inj_id, faults, accuracy) in enumerate(sorted(top_faults, reverse=True), start=1):
            fault_desc = ", ".join([f"{fault.layer_name}[{fault.tensor_index}] bit {fault.bits[0]}" for fault in faults])
            f.write(f"#{rank:3d} | Injection {inj_id:6d} | FR={failure_rate:.4f} | Acc={accuracy:.4f} | {fault_desc}\n")

    print(f"\nSaved top 100 critical faults to {output_path}")
    duration = time.time() - start_time
    print(f"\nAnalysis completed in {duration/60:.2f} minutes.")

if __name__ == "__main__":
    run_online_fault_injection_minimal()
