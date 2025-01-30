#!/bin/bash

# Define your choices array
choices=("mnist" "cifar10" "cifar100" "fashion_mnist" "skin_cancer")

# Loop through each choice and run the command with conditional GPU parameter
for choice in "${choices[@]}"; do
    python3 train_and_test.py --type "$choice" --gpu 0
done
