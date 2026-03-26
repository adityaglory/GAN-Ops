import os
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
DATASET_DIR = "dataset_images"
NUM_CLASSES = 3          # We will create 3 dummy classes
IMAGES_PER_CLASS = 50    # 50 images for each class
IMG_SIZE = 64
CHANNELS = 3             # 3 for RGB, 1 for Grayscale

def create_dummy_images():
    """
    Creates a directory structure and generates random noise images 
    to simulate a real image dataset for testing the training pipeline.
    """
    print(f"Starting dummy data generation in directory: '{DATASET_DIR}'...")
    
    # Create the main dataset directory if it doesn't exist
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        
    for class_idx in range(NUM_CLASSES):
        # Create a folder for each class (e.g., class_0, class_1)
        class_folder = os.path.join(DATASET_DIR, f"class_{class_idx}")
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
            
        print(f"Generating {IMAGES_PER_CLASS} images for {class_folder}...")
        
        for img_idx in range(IMAGES_PER_CLASS):
            # Generate random pixel values between 0 and 255
            if CHANNELS == 3:
                pixel_data = np.random.randint(0, 256, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            else:
                pixel_data = np.random.randint(0, 256, (IMG_SIZE, IMG_SIZE), dtype=np.uint8)
                
            # Convert numpy array to PIL Image
            img = Image.fromarray(pixel_data)
            
            # Save the image
            img_filename = f"dummy_{img_idx:03d}.png"
            img_path = os.path.join(class_folder, img_filename)
            img.save(img_path)
            
    print("Dummy dataset generation completed successfully!")

if __name__ == "__main__":
    create_dummy_images()