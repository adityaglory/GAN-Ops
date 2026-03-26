import numpy as np
import tensorflow as tf
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

class FIDEvaluator:
    def __init__(self, input_shape=(64, 64, 3)):
        print("Initializing FID Evaluator (Loading InceptionV3)...")
        # Load InceptionV3 without the classification head
        self.inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=input_shape)
        
    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2):
        """Calculates the Frechet Distance between two multivariate Gaussians."""
        # Calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2)**2.0)
        # Calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        # Check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        # Calculate score
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def get_fid_score(self, real_images, fake_images):
        """
        Takes batches of real and fake images (scaled [-1, 1]), 
        resizes them to 75x75 (minimum for Inception), and calculates FID.
        """
        # Resize images because InceptionV3 requires at least 75x75
        real_images = tf.image.resize(real_images, (75, 75))
        fake_images = tf.image.resize(fake_images, (75, 75))
        
        # Convert from [-1, 1] to [0, 255], then apply Inception preprocessing
        real_images = preprocess_input((real_images + 1.0) * 127.5)
        fake_images = preprocess_input((fake_images + 1.0) * 127.5)

        # Extract features
        act1 = self.inception_model.predict(real_images, verbose=0)
        act2 = self.inception_model.predict(fake_images, verbose=0)

        # Calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

        # Calculate FID
        fid = self.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        return fid