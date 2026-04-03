# GAN-Ops
# End-to-End MLOps Pipeline

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange.svg)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue.svg)](https://www.docker.com/)

An end-to-end Machine Learning Operations (MLOps) pipeline that trains a custom Conditional Wasserstein GAN with Gradient Penalty (WGAN-GP) to synthesize real-world flower textures, deployed as a containerized REST API.

## Project Highlights
* **Automated Data Ingestion:** Scripts to dynamically fetch, extract, and clean high-resolution datasets from Google's Cloud CDN.
* **Advanced Data Engineering:** CPU-bound data augmentation pipeline utilizing `albumentations` to maximize dataset variance.
* **Mathematical Stability:** Custom WGAN-GP implementation featuring Layer Normalization and Resize-Convolutions to eliminate mode collapse and checkerboard artifacts.
* **Telemetry & Evaluation:** Custom InceptionV3-based Fréchet Inception Distance (FID) evaluator and regex-based live diagnostic dashboards.
* **Microservice Deployment:** Lightweight, production-ready inference server built with FastAPI and fully containerized using Docker to prevent environment mismatches.

## How to Run (Inference via Docker)
The deployment environment is completely isolated and requires zero local Python configuration.

1. **Build the Container:**
   ```bash
   docker build -t gan_ops .
   ```
2. **Start the Microservice:**
    ```bash
    docker run -p 8000:8000 gan_ops
    ```
3. **Generate an Image: Open your browser and navigate to:**
   ```bash
   http://localhost:8000/generate?flower_type=roses
   ```

## How to Train (Local Development)
To run the training pipeline from scratch:
1. Install dependencies: 
```bash
pip install -r requirements.txt
```
2. Fetch the dataset: 
```bash
python download_real_data.py
```
3. Launch stabilized training with live logging:
```bash
python -u main_train.py | tee training_log.txt
````
4. Generate live telemetry dashboards: 
```bash
python plot_metrics.py
```
