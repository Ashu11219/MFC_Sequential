# Image Classification Model using TensorFlow

This project demonstrates how to build, train, and evaluate a Convolutional Neural Network (CNN) for image classification using TensorFlow and Keras. The model is trained on the CIFAR-10 dataset, which contains 60,000 32x32 color images in 10 classes.

## Project Structure

The project consists of the following files:
1. **`image_classification.py`**: The main Python script containing the code to load the dataset, build the model, train it, and evaluate its performance.
2. **`requirements.txt`**: A file listing all the dependencies required to run the project.
3. **`README.md`**: This file, providing an overview of the project and instructions for setup and execution.

## How It Works

1. **Dataset**: The CIFAR-10 dataset is loaded using TensorFlow's built-in `keras.datasets` module. The images are normalized, and the labels are one-hot encoded.
2. **Model**: A Sequential CNN model is built with the following layers:
   - Two Convolutional layers with MaxPooling and Dropout.
   - A Fully Connected (Dense) layer with Dropout.
   - An Output layer with Softmax activation for multi-class classification.
3. **Training**: The model is trained for 20 epochs using the Adam optimizer and categorical cross-entropy loss.
4. **Evaluation**: The model's performance is evaluated on the test set, and the accuracy is printed.
5. **Visualization**: Training and validation accuracy/loss are plotted using Matplotlib.
6. **Saving and Loading**: The trained model is saved to a file (`image_classification_model.h5`) and can be loaded later for inference.

## Installation

All the necessary dependencies are listed in the `requirements.txt` file. To install them, run the following command:

```bash
pip install -r requirements.txt
