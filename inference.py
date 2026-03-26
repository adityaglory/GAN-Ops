# inference.py
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def generate_and_save_plot():
    print("Loading the best WGAN-GP model...")
    model_path = 'best_cgan_wgan_gp.keras'
    
    try:
        generator = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    num_classes = 3
    samples_per_class = 4
    latent_dim = 100
    class_names = ['Daisy', 'Dandelion', 'Roses']

    # Create a figure to hold our image grid
    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(10, 8))
    fig.suptitle('WGAN-GP Generated Flowers (Early Training)', fontsize=16)

    print("Generating images...")
    for class_id in range(num_classes):
        # 1. Create noise and label vectors
        noise = tf.random.normal([samples_per_class, latent_dim])
        labels = tf.constant([class_id] * samples_per_class, dtype=tf.int32)
        
        # 2. Generate the images
        generated_images = generator([noise, labels], training=False)
        
        # 3. Post-process: Rescale from [-1, 1] to [0, 1] for matplotlib
        generated_images = (generated_images + 1.0) / 2.0
        generated_images = tf.clip_by_value(generated_images, 0.0, 1.0).numpy()
        
        # 4. Plot them in the grid
        for i in range(samples_per_class):
            ax = axes[class_id, i]
            ax.imshow(generated_images[i])
            ax.axis('off')
            if i == 0:
                ax.set_title(class_names[class_id])

    plt.tight_layout()
    output_filename = 'generated_samples.png'
    plt.savefig(output_filename)
    print(f"Success! Open '{output_filename}' to see what the model learned.")

if __name__ == "__main__":
    generate_and_save_plot()