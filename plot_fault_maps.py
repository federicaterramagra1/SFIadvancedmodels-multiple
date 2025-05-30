import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import io
import os
import SETTINGS

# Dimensioni note dei layer per SimpleMLP
LAYER_SHAPES = {
    'fc1': (6, 4),  # output x input
    'fc2': (2, 6)
}

# Carica top 100 fault più critici
N = SETTINGS.NUM_FAULTS_TO_INJECT
prefix = f"N{N}_"
top_faults_path = f"{prefix}top100_online_faults.csv"
output_dir = f"{prefix}fault_maps"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(top_faults_path)

# Funzione per generare immagine combinata fc1 + fc2 con didascalia (in memoria)
cmap_individual = plt.cm.Greys
cmap_individual.set_over('red')

def generate_combined_fault_image(fc1_map, fc2_map, label_text):
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))

    axs[0].imshow(fc1_map, cmap=cmap_individual, vmin=0, vmax=1.1)
    axs[0].set_title("fc1")
    axs[0].set_xticks(np.arange(fc1_map.shape[1]))
    axs[0].set_yticks(np.arange(fc1_map.shape[0]))

    axs[1].imshow(fc2_map, cmap=cmap_individual, vmin=0, vmax=1.1)
    axs[1].set_title("fc2")
    axs[1].set_xticks(np.arange(fc2_map.shape[1]))
    axs[1].set_yticks(np.arange(fc2_map.shape[0]))

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()

    buf.seek(0)
    image = Image.open(buf)
    width, height = image.size
    new_img = Image.new("RGB", (width, height + 30), (255, 255, 255))
    new_img.paste(image, (0, 0))
    draw = ImageDraw.Draw(new_img)
    font = ImageFont.load_default()
    draw.text((10, height + 5), label_text, fill=(0, 0, 0), font=font)
    return new_img

# Funzione per creare collage da immagini PIL
def create_collage(images, collage_path, grid=(5, 5), size=(300, 150), padding=10):
    width = grid[0] * size[0] + (grid[0] + 1) * padding
    height = grid[1] * size[1] + (grid[1] + 1) * padding
    collage = Image.new('RGB', (width, height), (255, 255, 255))

    for i, img in enumerate(images):
        row, col = divmod(i, grid[0])
        x = padding + col * (size[0] + padding)
        y = padding + row * (size[1] + padding)
        resized = img.resize(size)
        collage.paste(resized, (x, y))

    collage.save(collage_path)

# Genera le immagini combinate e crea collage ogni 25
current_batch = []
batch_size = 25
batch_index = 1

for idx, row in df.iterrows():
    inj_id = row['Injection']
    layers = eval(row['Layer'])
    indices = eval(row['TensorIndex'])

    fc1_map = np.zeros(LAYER_SHAPES['fc1'])
    fc2_map = np.zeros(LAYER_SHAPES['fc2'])

    fc1_indices = []
    fc2_indices = []

    for layer, idx in zip(layers, indices):
        r, c = eval(idx)
        if layer == 'fc1':
            fc1_map[r, c] = 2
            fc1_indices.append(f"({r},{c})")
        elif layer == 'fc2':
            fc2_map[r, c] = 2
            fc2_indices.append(f"({r},{c})")

    label = f"fc1: {', '.join(fc1_indices)} | fc2: {', '.join(fc2_indices)}"
    img = generate_combined_fault_image(fc1_map, fc2_map, label)
    current_batch.append(img)

    if len(current_batch) == batch_size:
        collage_file = os.path.join(output_dir, f"collage_batch_{batch_index}.png")
        create_collage(current_batch, collage_file)
        current_batch = []
        batch_index += 1

# Salva eventuale batch incompleto
if current_batch:
    collage_file = os.path.join(output_dir, f"collage_batch_{batch_index}.png")
    create_collage(current_batch, collage_file)
