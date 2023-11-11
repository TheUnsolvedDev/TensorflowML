#!/bin/bash

# Activate the TF_GPU conda environment
source activate TF_GPU

if ! python3 -c "import tensorflow as tf" &>/dev/null; then
    echo "Tensorflow is not installed. Installing..."
    pip3 install tensorflow[and-cuda]
    echo "Tensorflow has been installed successfully."
else
    echo "Tensorflow is installed"
fi
python3 -c "import tensorflow as tf; print('Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))"

if ! python3 -c "import matplotlib.pyplot as plt" &>/dev/null; then
    echo "matplotlib is not installed. Installing..."
    pip3 install matplotlib
    echo "matplotlib has been installed successfully."
else
    echo "matplotlib is installed"
fi

if ! python3 -c "import tqdm" &>/dev/null; then
    echo "tqdm is not installed. Installing..."
    pip3 install tqdm
    echo "tqdm has been installed successfully."
else
    echo "tqdm is installed"
fi

if ! python3 -c "import tensorflow_datasets" &>/dev/null; then
    echo "tensorflow_datasets is not installed. Installing..."
    pip3 install tensorflow_datasets
    echo "tensorflow_datasets has been installed successfully."
else
    echo "tensorflow_datasets is installed"
fi

if ! python3 -c "import cv2" &>/dev/null; then
    echo "OpenCV is not installed. Installing..."
    pip3 install opencv-python
    echo "OpenCV has been installed successfully."
else
    echo "OpenCV is installed"
fi

if ! python3 -c "import tensorflow_probability" &>/dev/null; then
    echo "tensorflow_probability is not installed. Installing..."
    pip3 install tensorflow_probability
    echo "tensorflow_probability has been installed successfully."
else
    echo "tensorflow_probability is installed"
fi

if ! python3 -c "import silence_tensorflow" &>/dev/null; then
    echo "silence_tensorflow is not installed. Installing..."
    pip3 install silence_tensorflow
    echo "silence_tensorflow has been installed successfully."
else
    echo "silence_tensorflow is installed"
fi

if ! python3 -c "import tensorflow_model_optimization as tfmot" &>/dev/null; then
    echo "tensorflow_model_optimization is not installed. Installing..."
    pip3 install tensorflow_model_optimization
    echo "tensorflow_model_optimization has been installed successfully."
else
    echo "tensorflow_model_optimization is installed"
fi

echo "Env Checked go to go!!"
conda deactivate
sleep 5
clear

