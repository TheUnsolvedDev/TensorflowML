#!/bin/bash
conda create -n TF_GPU python=3.11
conda activate TF_GPU
pip3 install tensorflow[and-cuda]
pip3 install tensorflow_datasets
pip3 install tqdm
pip3 install numpy 
pip3 install matplotlib 
pip3 install tensorflow_probability
pip3 install silence_tensorflow
