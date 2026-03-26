import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all logs, 1=info, 2=warnings, 3=errors only
import tensorflow as tf
from cgan_model import build_generator, build_discriminator
from train import train_cgan

# --- HYPERPARAMETERS ---
DATA_DIR = "dataset_real_images"
IMG_SIZE = 64
CHANNELS = 3 # Change to 1 if your images are Grayscale
BATCH_SIZE = 32
LATENT_DIM = 100
EPOCHS = 500

def main():
    # 1. Check if data folder exists
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Folder '{DATA_DIR}' not found!")
        print("Please create the folder and fill it with image data per class.")
        return

    # 2. Process Data
    dataset, num_classes = load_and_preprocess_data(DATA_DIR, (IMG_SIZE, IMG_SIZE), BATCH_SIZE)

    # 3. Build Model Architecture
    print("Building Generator and Discriminator models...")
    generator = build_generator(IMG_SIZE, CHANNELS, LATENT_DIM, num_classes)
    discriminator = build_discriminator(IMG_SIZE, CHANNELS, num_classes)

    # 4. Start Training
    train_cgan(dataset, generator, discriminator, EPOCHS, LATENT_DIM)

if __name__ == "__main__":
    main()