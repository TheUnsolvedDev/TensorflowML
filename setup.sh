#!/bin/bash
python -m venv GPU_tf
source GPU_tf/bin/activate
pip3 install tensorflow[and-cuda]
pip3 install tensorflow_datasets
pip3 install tqdm
pip3 install numpy 
pip3 install matplotlib 
pip3 install tensorflow_probability