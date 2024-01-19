# Image-Denoising-using-autoencoder
# MNIST Neural Network Project

A project that utilizes the MNIST dataset to create two neural network models: one for classification of handwritten digits and another for de-noising input data using an autoencoder. These models are then combined into a composite model for improved performance.

## Table of Contents

- [MNIST Neural Network Project](#mnist-neural-network-project)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [Acknowledgments](#acknowledgments)
  - [Contact](#contact)

## Description

This project focuses on the MNIST dataset, consisting of 60,000 training examples and 10,000 test examples of handwritten digits. The main objectives include:

1. **Classification Model**: A neural network trained to classify handwritten digits.
2. **Autoencoder Model**: Used to de-noise input data by reducing dimensionality and focusing on important features.
3. **Composite Model**: Combining the classification and autoencoder models into a single pipeline for improved performance.

The training process involves normalizing values, reshaping examples, and artificially adding noise to training and test set examples.

## Features

- Classification of handwritten digits
- De-noising input data using an autoencoder
- Composite model combining classification and de-noising

## Installation

Guide users on how to install and set up your project. Include any dependencies and system requirements.

import numpy as np

from tensorflow.keras.datasets import mnist

from matplotlib import pyplot as plt

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback

from tensorflow.keras.utils import to_categorical


# Example usage
python train_classification_model.py
python train_autoencoder_model.py
python composite_model.py

Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:

Fork the project
Create a new branch
Make your changes
Submit a pull request


Acknowledgments
MNIST dataset contributors

Contact
For any questions or feedback, please contact:

Aditya Aryan Verma
adityaverma655@gmail.com



