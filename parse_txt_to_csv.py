import re
import csv

input_path = "top100_faults_N4.txt"
output_path = "N4_top100_online_faults.csv"

def parse_fault_line(line):
    # Estrai injection ID, failure rate e accuracy
    match = re.match(r"#\s*\d+\s+\|\s+Injection\s+(\d+)\s+\|\s+FR=([\d.]+)\s+\|\s+Acc=([\d.]+)\s+\|\s+(.+)", line)
    if not match:
        return None

    inj_id = int(match.group(1))
    fr = float(match.group(2))
    acc = float(match.group(3))
    faults_str = match.group(4)

    # Estrai triplette (layer, tensor index, bit)
    pattern = r"(\w+)\[\((\d+),\s*(\d+)\)\]\s+bit\s+(\d+)"
    matches = re.findall(pattern, faults_str)

    layers = []
    tensor_indices = []
    bits = []

    for layer, r, c, bit in matches:
        layers.append(layer)
        tensor_indices.append(f"({r},{c})")
        bits.append(int(bit))

    return {
        "Injection": inj_id,
        "Layer": layers,
        "TensorIndex": tensor_indices,
        "Bit": bits,
        "FailureRate": fr,
        "Accuracy": acc
    }

# Scrittura CSV
with open(input_path, "r") as f_in, open(output_path, "w", newline="") as f_out:
    writer = csv.DictWriter(f_out, fieldnames=["Injection", "Layer", "TensorIndex", "Bit", "FailureRate", "Accuracy"])
    writer.writeheader()
    for line in f_in:
        if not line.strip().startswith("#"):
            continue
        parsed = parse_fault_line(line)
        if parsed:
            writer.writerow(parsed)

print(f" CSV salvato in: {output_path}")
