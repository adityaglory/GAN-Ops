# download_real_data.py
import os
import shutil
import pathlib
import tensorflow as tf
from PIL import Image

def setup_fast_real_dataset():
    print("Downloading 'Flowers' dataset from Google's high-speed CDN...")
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    
    # Download and extract
    data_dir = tf.keras.utils.get_file(origin=dataset_url, 
                                       fname='flower_photos', 
                                       untar=True)
    data_dir = pathlib.Path(data_dir)
    
    # If get_file returns the tarball path instead of the directory, adjust it
    if data_dir.is_file() or data_dir.suffix == '.tar.gz':
        data_dir = data_dir.parent / "flower_photos"
        
    print(f"Searching for images in: {data_dir}")

    target_classes = ['daisy', 'dandelion', 'roses']
    IMAGES_PER_CLASS = 300 
    
    base_dir = "dataset_real_images"
    
    # Clean up the old directory if it exists
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    
    for class_id, class_name in enumerate(target_classes):
        target_class_dir = os.path.join(base_dir, f"class_{class_id}_{class_name}")
        os.makedirs(target_class_dir, exist_ok=True)
        
        # Use recursive glob to find images anywhere inside the extracted folder
        all_images = list(data_dir.rglob('*.jpg'))
        
        # Filter images that belong to the current class folder name
        class_images = [img for img in all_images if class_name in img.parts]
        
        # Limit to the requested number of images
        class_images = class_images[:IMAGES_PER_CLASS]
        
        print(f"Processing and resizing {len(class_images)} images for class '{class_name}'...")
        
        for i, img_path in enumerate(class_images):
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((64, 64), Image.LANCZOS)
                img.save(os.path.join(target_class_dir, f"{class_name}_{i:03d}.png"))
            except Exception as e:
                print(f"Skipping corrupted image {img_path}: {e}")
                
    print(f"Real dataset successfully prepared in '{base_dir}'!")

if __name__ == "__main__":
    setup_fast_real_dataset()