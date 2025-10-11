<!-- # 🩻 Class-Dependent Miscalibration Severely Degrades Selective Prediction in Multimodal Clinical Prediction Models -->


Table of contents
=================

  * [Background](#Background)
  * [Directory overview](#Directory-overview)
  * [Getting started](#Getting-started)
  * [License](#License)


Background
============
As artificial intelligence systems transition from research to clinical deployment, ensuring their reliability becomes critical for clinical decision-making tasks, as incorrect predictions can have serious consequences. 
Deploying AI in healthcare therefore requires prediction systems with robust safeguards against error, such as selective prediction, where uncertain predictions are deferred to human experts for review. 
In this study, we carefully evaluate the reliability of uncertainty-based selective prediction for multilabel clinical condition classification using multimodal data. 
Our findings show that models often exhibit severe class-dependent miscalibration causing predictive performance to degrade under uncertainty-guided selective prediction---attributing high uncertainty to correct predictions and low uncertainty to incorrect predictions. 
Our evaluation highlights fundamental shortcomings of commonly used evaluation metrics for clinical AI. 
To address these shortcomings, we propose practical recommendations for calibration-aware model assessment and selective prediction design, offering a pathway to safer, more reliable AI systems that clinicians and patients can trust.

Directory overview
====================================

### MedFuse/
- Contains the main MedFuse model architectures.

### configs/
- Configuration files for different training methods, unimodal and multimodal as well as hyperparameters for the group aware priors framework.

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
- Main script.
- **Functionality**:
  - **Data Loading**: Prepares data for training and evaluation.
  - **Model Initialization**: Sets up the LSTM and ResNet models.
  - **Training Loops**: Core training routines.
  - **Evaluation**: Performance assessment of trained models.
  - **Optimizers**: Configuration of optimization algorithms.
  - **Data Extraction During Inference**: Extracts data for analysis during model inference.

Getting started
====================================

To get started, follow the instructions below.

### Prerequisites

- Python 3.8+
- Conda environment (recommended)

### Installation

1. Clone the repository:
   ```sh
   git clone https://anonymous.4open.science/r/medcalibration-B187/README.md

2. Clone the repository:

    ```conda env create -f environment.yml
    conda activate uq-wq

3. Refer to **shell_scripts/** and **job_files/** folders for specific cases of model training


License
====================================
This project is licensed under the MIT License. See the LICENSE file for details.