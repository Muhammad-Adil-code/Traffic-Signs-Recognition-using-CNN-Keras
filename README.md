# Traffic Signs Recognition using CNN and Keras

This project focuses on building a deep learning model to recognize and classify traffic signs from images. Leveraging a Convolutional Neural Network (CNN), the model is trained to identify 58 different classes of traffic signs, a critical task for developing autonomous driving systems and enhancing road safety.



## Table of Contents
- [Project Summary](#project-summary)
- [The Problem](#the-problem)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Workflow](#project-workflow)
  - [1. Data Loading and Visualization](#1-data-loading-and-visualization)
  - [2. Data Preparation and Augmentation](#2-data-preparation-and-augmentation)
  - [3. Model Architecture](#3-model-architecture)
  - [4. Model Training](#4-model-training)
  - [5. Model Evaluation](#5-model-evaluation)
- [Conclusion](#conclusion)

## Project Summary
This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify traffic signs. The workflow begins by loading a dataset containing 4,170 images across 58 different classes. The data is first visualized to understand its structure and content.

Next, the dataset is prepared for training by splitting it into training and validation sets. Data augmentation techniques, such as random flips, rotations, and zooms, are applied to the training data to increase its diversity and prevent overfitting.

A sequential CNN model is then constructed with four convolutional layers, each followed by a max-pooling layer to extract features from the images. The model is compiled with an 'adam' optimizer and trained for up to 50 epochs, using `EarlyStopping` to halt training when validation performance ceases to improve. Finally, the model's performance is evaluated by plotting its accuracy and loss curves over the training process.

## The Problem
Road accidents are a major concern, with incidents often increasing due to factors like overspeeding or poor visibility of traffic signs, especially in adverse weather conditions like winter fog. An automated system capable of accurately recognizing traffic signs can significantly enhance driver safety and is a foundational component for autonomous vehicles. This project aims to build such a system.

## Dataset
The project utilizes a traffic sign dataset that is composed of:
- **4,170 images** in total.
- **58 different classes** of traffic signs.
- A `labels.csv` file that maps class IDs to their corresponding names (e.g., 'Stop', 'Speed limit 50km/h').

The dataset is loaded and split into a training set (3,336 images) and a validation set (834 images).

## Installation
To run this project, you need Python and the following libraries. It is recommended to set up a virtual environment.

1.  **Clone a repository or set up a project folder.**
2.  **Install the required libraries:**
    ```bash
    pip install tensorflow pandas numpy matplotlib opencv-python scikit-learn
    ```
3.  **Download the dataset:**
    Acquire the `traffic-sign-dataset-classification.zip` file and place it in your project directory.

## Project Workflow

### 1. Data Loading and Visualization
The process starts by importing necessary libraries and unzipping the dataset. The `labels.csv` file is loaded into a pandas DataFrame to map class IDs to human-readable names. Images from different classes are visualized to get a feel for the dataset.

### 2. Data Preparation and Augmentation
- **Data Splitting:** The `image_dataset_from_directory` function from Keras is used to load the images and split them directly into training and validation sets (80-20 split). The images are resized to 224x224 pixels.
- **Data Augmentation:** To make the model more robust and prevent it from memorizing the training data, data augmentation is applied. This involves creating new training samples by applying random transformations to the existing images, such as:
  - Random horizontal and vertical flips.
  - Random rotations.
  - Random zooming.

### 3. Model Architecture
A sequential CNN model is designed with the following layers:
- **Data Augmentation Layer:** The augmentation pipeline defined earlier.
- **Rescaling Layer:** Normalizes pixel values from the `[0, 255]` range to `[0, 1]`.
- **Four Convolutional Blocks:**
  - `Conv2D` layer with ReLU activation to learn features.
  - `MaxPooling2D` layer to downsample the feature maps, reducing computational complexity and making the learned features more robust.
- **Flatten Layer:** Converts the 2D feature maps into a 1D vector.
- **Fully Connected Layers (Dense):**
  - Two `Dense` layers with ReLU activation to learn high-level patterns.
  - A `Dropout` layer is added between them to reduce overfitting.
- **Output Layer:** A `Dense` layer with a `softmax` activation function to output a probability distribution over the 58 classes.

The model is compiled using:
- **Loss Function:** `SparseCategoricalCrossentropy`, suitable for multi-class classification where labels are integers.
- **Optimizer:** `Adam`, an efficient and popular optimization algorithm.
- **Metrics:** `accuracy` is tracked to monitor performance.

### 4. Model Training
The model is trained using the `.fit()` method, passing the training and validation datasets.
- **Epochs:** Set to 50, allowing the model sufficient iterations to learn from the data.
- **Callbacks:** An `EarlyStopping` callback is used to monitor the validation loss (`val_loss`). If the validation loss does not improve for 5 consecutive epochs (`patience=5`), the training process is halted to prevent overfitting and save time.

### 5. Model Evaluation
After training, the model's performance is evaluated by plotting the training and validation history for both accuracy and loss.
- **Accuracy Plot (`accuracy` vs. `val_accuracy`):** Shows how well the model is learning to classify the images correctly on both seen and unseen data.
- **Loss Plot (`loss` vs. `val_loss`):** Shows the error of the model. A decreasing loss indicates that the model is learning effectively.

The final plots demonstrate that the CNN model performs very well, achieving high accuracy on the validation set, indicating its effectiveness in recognizing traffic signs.

## Conclusion
This project successfully demonstrates the power of Convolutional Neural Networks for image classification tasks. By building and training a CNN, we have created a high-performance model capable of accurately identifying a wide variety of traffic signs. Such a system serves as a crucial building block for advanced driver-assistance systems (ADAS) and fully autonomous vehicles, contributing to safer roads for everyone.
