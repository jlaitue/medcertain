# 🩻 MedCertAIn: Uncertainty Quantification Framework with Informative Multimodal Priors Enhances Reliability of AI-based In-Hospital Mortality Prediction

### Release

- [10/2024] **MedCertAIn** (Initial) base code is released for submission to peer-reviewed journal.

Table of contents
=================

  * [Background](#Background)
  * [Directory overview](#Directory-overview)
  * [Getting started](#Getting-started)
  * [License](#License)


Background
============
Integrating AI in clinical decision support systems aims to enhance patient care. However, a major challenge is the lack of principled uncertainty quantification in machine learning models, especially in multimodal learning, which limits real-world deployment. We introduce MedCertAIn, a framework using tailored data-driven prior distributions over neural network parameters to improve uncertainty quantification. Implemented using JAX, MedCertAIn leverages the MIMIC-IV and MIMIC-CXR datasets to predict patient mortality in the ICU. Our results show that MedCertAIn significantly improves predictive performance and uncertainty metrics, outperforming current methods and demonstrating its potential for more reliable AI-driven healthcare applications.

Directory overview
====================================

### MedFuse/
- Contains the main MedFuse model architectures.

### configs/
- Configuration files for different training methods, unimodal and multimodal as well as hyperparameters for the uncertainty quantification framework.

### processing_scripts/
- Scripts for processing images, datasets, and evaluation metrics.

### shell_scripts/ and job_files/
- Scripts for submitting batch jobs to an HPC cluster.

### arguments.py
- Parses arguments sent through the terminal.

### base_architectures.py and base_datasets.py
- Base code to loading and processing datasets and model weights.

### functions.py
- Contains various utility functions used throughout the code for calculating performance metrics and custom loss functions used in model training.

### trainer.py
- Main script of the MedCertAIn framework.
- **Functionality**:
  - **Data Loading**: Prepares data for training and evaluation.
  - **Model Initialization**: Sets up the LSTM and ResNet models.
  - **Training Loops**: Core training routines.
  - **Evaluation**: Performance assessment of trained models.
  - **Optimizers**: Configuration of optimization algorithms.
  - **Data Extraction During Inference**: Extracts data for analysis during model inference.

Getting started
====================================

To get started with the MedCertAIn framework, follow the instructions below.

### Prerequisites

- Python 3.8+
- Conda environment (recommended)

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/MedCertAIn.git
   cd MedCertAIn

2. Clone the repository:

    ```conda env create -f environment.yml
    conda activate MedCertAIn

3. Refer to **shell_scripts/** and **job_files/** folders for specific cases of model training

*Note: we are currently in the process of preparing model weights for all trained models (i.e. unimodal and multimodal) adjusting the shell scripts for easier implementation.*


License
====================================
This project is licensed under the MIT License. See the LICENSE file for details.