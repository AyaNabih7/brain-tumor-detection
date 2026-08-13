# Brain Tumor Detection

A collection of Jupyter Notebooks and analysis code for detecting brain tumors from MRI images using machine learning and deep learning techniques. This repository contains exploratory data analysis, preprocessing, model training and evaluation notebooks, plus utilities to reproduce results and extend experiments.

## Table of Contents
- [Project overview](#project-overview)
- [Contents](#contents)
- [Dataset](#dataset)
- [Method overview](#method-overview)
- [Requirements](#requirements)
- [Getting started](#getting-started)
- [Notebook workflow](#notebook-workflow)
- [Results & evaluation](#results--evaluation)
- [Reproducing experiments](#reproducing-experiments)
- [Contributing](#contributing)
- [License & attribution](#license--attribution)
- [Contact](#contact)

## Project overview
This project explores automated brain tumor detection from MRI scans. The notebooks walk through data preparation, visualization, model development (custom CNNs and transfer learning), training, evaluation, and basic model export steps. The goal is to provide a reproducible pipeline and examples that can be adapted to other medical imaging classification tasks.

## Contents
- Jupyter Notebooks (.ipynb) with end-to-end experiments:
  - data_exploration.ipynb — dataset inspection and visualizations
  - preprocessing.ipynb — image preprocessing, augmentation, and dataset pipeline construction
  - model_training.ipynb — model architectures, training, checkpoints, and logging
  - evaluation.ipynb — metrics, confusion matrices, ROC curves, and error analysis
- utils/ — helper functions (data loaders, augmentations, metrics)
- requirements.txt — (optional) Python package dependencies
- README.md — this file

(If notebook filenames differ in the repository, please update the list above to match.)

## Dataset
This repository expects a labeled MRI brain image dataset organized in a clear folder structure. Example structure:

- data/
  - train/
    - tumor/
    - no_tumor/
  - val/
    - tumor/
    - no_tumor/
  - test/
    - tumor/
    - no_tumor/

If using a public dataset (for example, "Brain MRI Images for Brain Tumor Detection" from Kaggle or similar), extract the dataset and place it under `data/` or update notebook paths to point to your dataset location. For large datasets, keep raw data outside the repo and configure notebooks to read from a local path or environment variable.

## Method overview
Typical steps included in the notebooks:
1. Data loading and inspection (image counts, class distribution).
2. Preprocessing: resizing, normalization, optional skull-stripping or intensity normalization.
3. Augmentation: random rotations, flips, shifts, brightness/contrast adjustments.
4. Model training:
   - Custom convolutional neural networks trained from scratch.
   - Transfer learning using pre-trained backbones (e.g., ResNet, EfficientNet) with fine-tuning.
5. Evaluation: accuracy, precision, recall, F1-score, ROC/AUC, confusion matrices.
6. Model export and simple inference scripts/notebooks for predictions.

## Requirements
Recommended Python packages (use a virtualenv or conda environment). If you have a requirements.txt, install with the command below; otherwise, create one based on the list.

Install:
```bash
pip install -r requirements.txt
```

Common packages:
- Python 3.8+
- jupyterlab or notebook
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow (or torch + torchvision if using PyTorch)
- opencv-python
- pillow
- tqdm
- scikit-image

Tip: For reproducibility, pin package versions in requirements.txt or provide an environment.yml.

## Getting started
1. Clone the repository:
   ```bash
   git clone https://github.com/AyaNabih7/brain-tumor-detection.git
   cd brain-tumor-detection
   ```
2. Create and activate a virtual environment, then install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux / macOS
   .venv\Scripts\activate      # Windows (PowerShell)
   pip install -r requirements.txt
   ```
3. Launch Jupyter:
   ```bash
   jupyter lab
   ```
4. Update dataset paths in the first cells of each notebook to point to your local data.
5. Run notebooks in order: exploration → preprocessing → training → evaluation.

If you have a compatible GPU and CUDA installed, configure TensorFlow or PyTorch to use it for faster training.

## Notebook workflow
- data_exploration.ipynb: Inspect image samples, class balance, and statistics.
- preprocessing.ipynb: Implement resizing, normalization, dataset splits, and augmentation pipelines.
- model_training.ipynb: Define models, training loop, checkpointing, and logging (TensorBoard or other).
- evaluation.ipynb: Compute metrics, plot ROC curves/confusion matrices, and review misclassified samples.

Always record hyperparameters (learning rate, batch size, epochs, augmentation) to reproduce experiments and compare runs.

## Results & evaluation
The evaluation notebook computes and visualizes:
- Confusion matrix
- Precision, recall, F1-score per class
- ROC curves and AUC
- Example predictions and visual explanations (Grad-CAM or saliency maps if implemented)

Include saved plots and metrics in a results/ or reports/ folder when sharing experiments.

## Reproducing experiments
- Set a global random seed in notebooks (numpy, Python random, framework-specific seeds).
- Log experiments (TensorBoard, Weights & Biases, or CSV logs).
- Save model checkpoints and final weights with clear naming (date, model, dataset, hyperparams).
- Document dataset version and preprocessing steps applied.

## Troubleshooting tips
- If training diverges, try reducing learning rate, increasing batch size (if memory permits), or strengthening regularization/augmentation.
- For class imbalance, use stratified splits, class weights, or oversampling.
- Check input image channels and scaling (0–1 vs 0–255) if validation performance is poor.

