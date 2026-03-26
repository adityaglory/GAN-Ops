# main.py
from fastapi import FastAPI, HTTPException
import tensorflow as tf
import numpy as np
import base64
import io
from PIL import Image

app = FastAPI(
    title="cGAN Augmentation API", 
    description="End-to-End API for generating synthetic data using Conditional GANs"
)

# --- Configuration ---
LATENT_DIM = 100
MODEL_PATH = "cgan_generator.h5" 
NUM_CLASSES = 3 # Adjust this if you train with more classes later
IMG_SIZE = 64
CHANNELS = 3

# Load the model globally when the API starts
try:
    print(f"Loading model from {MODEL_PATH}...")
    generator = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    generator = None

@app.post("/generate")
async def generate_data(class_label: int, num_samples: int = 1):
    """
    Endpoint to generate synthetic images for a specific class.
    """
    if generator is None:
         raise HTTPException(status_code=500, detail="Generator model is not loaded on the server.")
         
    if num_samples < 1 or num_samples > 10:
        raise HTTPException(status_code=400, detail="Number of samples must be between 1 and 10.")
        
    if class_label < 0 or class_label >= NUM_CLASSES:
        raise HTTPException(status_code=400, detail=f"Invalid class_label. Must be between 0 and {NUM_CLASSES - 1}.")
    
    try:
        # 1. Create noise latent vector
        noise = tf.random.normal([num_samples, LATENT_DIM])
        
        # 2. Create class labels tensor
        labels = tf.constant([class_label] * num_samples, dtype=tf.int32)
        
        # 3. Generate images using the cGAN model
        # Our cGAN model expects a list of inputs: [noise, labels]
        generated_images = generator([noise, labels], training=False)
        
        # 4. Post-process images (Convert from [-1, 1] to [0, 255])
        generated_images = (generated_images + 1.0) * 127.5
        generated_images = tf.cast(generated_images, tf.uint8).numpy()
        
        # 5. Convert to Base64
        results = []
        for img_array in generated_images:
            if CHANNELS == 1:
                img_reshaped = img_array.reshape(IMG_SIZE, IMG_SIZE) 
                img = Image.fromarray(img_reshaped)
            else:
                img = Image.fromarray(img_array)
            
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            results.append({"image_base64": img_str, "class_label": class_label})
            
        return {
            "status": "success", 
            "message": f"{num_samples} images generated for class {class_label}", 
            "data": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error during generation: {str(e)}")