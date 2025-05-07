#!/bin/bash

# Define your choices array
choices=("skin_cancer" "cassava_leaf_disease" "chest_xray" "crop_disease" "mnist" "cifar10" "cifar100" "fashion_mnist")

# Loop through each choice and run the command with conditional GPU parameter
for choice in "${choices[@]}"; do
    python3 train_and_test.py --type "$choice" --gpu -1
done
