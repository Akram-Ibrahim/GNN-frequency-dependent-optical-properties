# GNN-Frequency-Dependent-Optical-Properties

This project leverages **Graph Neural Networks (GNNs)** to predict the **frequency-dependent dielectric function** from crystal structures. The data and model provided in this repository support the findings from the paper:  
**"Prediction of Frequency-Dependent Optical Spectrum for Solid Materials: A Multioutput and Multifidelity Machine Learning Approach"**  
[Read the paper here](https://pubs.acs.org/doi/10.1021/acsami.4c07328).

## Project Overview

The repository provides code and data for training and evaluating a GNN-based model that predicts the optical spectra of materials across various frequencies. The approach integrates multi-fidelity data and compares multi-output scaling/loss architectures to enhance the learning process of high-fidelity optical spectra.

### Key Features

- **Multi-Fidelity Learning**: Employs and compares fidelity embeddings and transfer learning to integrate data from both high- and low-fidelity sources.
- **Custom GNN Architecture**: Implements a custom architecture with MEGNet layers for effective feature extraction from crystal structures.
- **Custom Loss Functions**: Applies and compares tailored loss functions (MAE, KL divergence, Wasserstein) and scaling schemes (UnNorm, MaxNorm, AvgNorm) to minimize prediction errors for optical spectra.

## How to Use

### Environment Setup
Ensure you have the following dependencies installed:
- Python 3.x
- TensorFlow 2.x
- MEGNet
- Pymatgen/ASE (for structure handling)
- Optuna (for hyperparameter optimization)
- Seaborn, Matplotlib (for plotting)

### Model Training
Modify the input parameters/hyperparameters directly in the code as needed, and execute it to start training:
- `loss_weights`: Adjust in case of normalized spectrum learning.
- `embedding_dimensions` for atom, bond, and state features.
- `GNN layer sizes`, `Dropout rates`, `Learning rate`, and `Decay factor`, etc.


### Model Evaluation
The script automatically evaluates the model using a test set, plotting both training and validation losses over epochs. It also calculates custom metrics for component-wise errors across the predicted spectra and target norms.


## General Layout
<p align="center">
  <img src="https://github.com/user-attachments/assets/9807bda2-eab4-4d77-a2f7-22a00da6b05d" width="100%">
</p>

## GNN Predictions for the Imaginary Part of the Dielectric Function and the Absorption Coefficient
**Predictions of the imaginary part of the dielectric function and the absorption coefficient at meta-GGA MBJ functional accuracy.**
<p align="center">
  <img src="https://github.com/user-attachments/assets/be647881-7478-48a3-804e-84b201523e8c" width="100%">
</p>

## GNN Predictions for the Short-Circuit Current, Reverse Saturation Current, and Spectroscopic Limit of Maximum Efficiency
<p align="center">
  <img src="https://github.com/user-attachments/assets/51711136-1d67-4d8a-ba23-4368153d14b2" width="100%">
</p>



