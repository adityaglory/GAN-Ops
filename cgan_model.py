import tensorflow as tf
from tensorflow.keras import layers

def build_generator(img_size, channels, latent_dim, num_classes):
    label_input = layers.Input(shape=(1,), name='label_input')
    label_embedding = layers.Embedding(num_classes, 50)(label_input)
    label_dense = layers.Dense(8 * 8)(label_embedding)
    label_reshape = layers.Reshape((8, 8, 1))(label_dense)
    
    noise_input = layers.Input(shape=(latent_dim,), name='noise_input')
    noise_dense = layers.Dense(8 * 8 * 128)(noise_input)
    noise_reshape = layers.Reshape((8, 8, 128))(noise_dense)
    
    merged = layers.Concatenate()([noise_reshape, label_reshape])
    
    x = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same')(merged)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    
    x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    
    x = layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    
    output_image = layers.Conv2D(channels, kernel_size=4, padding='same', activation='tanh')(x)
    return tf.keras.Model([noise_input, label_input], output_image, name="generator")


def build_discriminator(img_size, channels, num_classes):
    label_input = layers.Input(shape=(1,), name='label_input')
    label_embedding = layers.Embedding(num_classes, 50)(label_input)
    label_dense = layers.Dense(img_size * img_size)(label_embedding)
    label_reshape = layers.Reshape((img_size, img_size, 1))(label_dense)
    
    image_input = layers.Input(shape=(img_size, img_size, channels), name='image_input')
    
    merged = layers.Concatenate()([image_input, label_reshape])
    
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same')(merged)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    
    x = layers.Conv2D(256, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    output_decision = layers.Dense(1)(x)
    return tf.keras.Model([image_input, label_input], output_decision, name="discriminator")