# Electrical & Mechanical Components — Image Classification

**Project:** Image classification of 7 categories of electrical & mechanical parts

**Categories:** `ac_motor`, `bevel_gear`, `crank_shaft`, `dc_motor`, `motor_winding`, `pistons`, `spur_gear`

**Overview**
This repository contains a deep learning project for classifying images of electrical and mechanical components. The images were collected and organized into categorized folders. The pipeline covers dataset preparation, model building (CNN + small ANN head), training, evaluation, test predictions, and a real-time demo using a Streamlit app. The dataset is hosted on Kaggle
(https://www.kaggle.com/datasets/venkatesh2410/images).

---

## Table of contents

1. [Project Structure](#project-structure)
2. [Dataset](#dataset)
3. [Setup & Requirements](#setup--requirements)
4. [Data Preparation](#data-preparation)
5. [Model](#model)
6. [Training](#training)
7. [Evaluation & Results](#evaluation--results)
8. [Inference & Real-time App (Streamlit)](#inference--real-time-app-streamlit)
9. [How to run](#how-to-run)
10. [Tips & Notes](#tips--notes)
11. [License & Contact](#license--contact)

---

## Dataset



---

## Data Preparation

1. Arrange images into a single `data/organized/<class_name>/` structure.
2. Run the provided split script (or use the notebook) to create `train/`, `val/`, `test/` subfolders. Typical splits: 70% train / 20% val / 10% test.

Example script (in `src/data_utils.py`) uses `ImageDataGenerator` for building generators. Key preprocessing steps:

* Resize images to a consistent shape (e.g. 224x224 or 256x256)
* Normalize pixel values (scale to `[0,1]`) or use standardized ImageNet means if using transfer learning
* Data augmentation (random rotations, flips, shifts) applied to the training generator only

---

## Model

This project uses a **CNN feature extractor + a small ANN head** to combine deep features.

---

## Training

**Sample training hyperparameters**

* Batch size: 16–64 (depending on GPU memory)
* Epochs: 25–60 (use early stopping)
* Optimizer: Adam with learning rate `1e-4` (or use AdamW/SGD for fine tuning)
* Loss: Categorical Crossentropy
* Metrics: Accuracy, Precision, Recall, F1 (compute at evaluation time)

Use callbacks: `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`.

* from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau.

## Evaluation & Results

* Evaluate on the held-out `test/` set with the saved best model.
* Compute confusion matrix and per-class precision/recall/F1 score using `sklearn.metrics`.

Include plots for training/validation loss and accuracy in the `notebooks/03_evaluation.ipynb`.

**If you want me to add a sample of results (confusion matrix and metrics), provide the model output or training logs and I will generate the plots and tables.**

---

## Inference & Real-time App (Streamlit)

A simple Streamlit app is included (`app/streamlit_app.py`) to demo real-time predictions. It loads the trained model and allows users to upload images or use webcam (if desired).

---

## Tips & Notes

* If classes are imbalanced, consider class weights or oversampling/augmentation.
* For faster iteration, use a smaller image size (e.g., 128x128) for experiments, then scale up for final training.
* If GPU memory is limited, reduce batch size or use mixed precision training.
* For deployment consider converting the model to TensorFlow SavedModel or ONNX for cross-platform usage.

---

## License

This project is released under the MIT License. Update `LICENSE` file accordingly.

---

## Contact

Maintainer: Venkatesh — venkateshvarada56@gmail.com.

---
