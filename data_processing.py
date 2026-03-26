# data_processing.py
import tensorflow as tf
import albumentations as A
import numpy as np

# 1. Define the Albumentations Pipeline
# These are standard, safe augmentations that prevent the Discriminator from memorizing
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

def albumentations_wrapper(image):
    """
    Applies albumentations to a single numpy image array.
    """
    # Albumentations expects uint8 data for colors
    aug_data = transform(image=image.astype(np.uint8))
    aug_img = aug_data["image"]
    
    # Normalize to [-1, 1] for the GAN's tanh activation
    aug_img = (aug_img.astype(np.float32) / 127.5) - 1.0
    return aug_img

def process_tf_image(image, label):
    """
    Bridges the TensorFlow graph with our Python/Numpy albumentations function.
    """
    # tf.numpy_function runs the python function inside the TF pipeline
    aug_img = tf.numpy_function(func=albumentations_wrapper, inp=[image], Tout=tf.float32)
    
    # tf.numpy_function loses the shape information, so we must restore it manually
    aug_img.set_shape((64, 64, 3)) 
    return aug_img, label

def load_and_preprocess_data(data_dir, img_size=(64, 64), batch_size=32):
    print(f"Loading and augmenting data from: {data_dir}...")
    
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='int',
        image_size=img_size,
        batch_size=batch_size, # This loads data in batches
        shuffle=True
    )
    
    class_names = dataset.class_names
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # 2. Apply the augmentations
    # We unbatch the dataset to apply the augmentations image-by-image, then re-batch
    dataset = dataset.unbatch()
    dataset = dataset.map(process_tf_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True) # drop_remainder stabilizes WGAN training
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, num_classes