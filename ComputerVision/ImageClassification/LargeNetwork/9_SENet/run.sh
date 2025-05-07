#!/bin/bash

# Define your choices array
choices=("mnist" "cifar10" "cifar100" "fashion_mnist")

# Loop through each choice and run the command with conditional GPU parameter
for choice in "${choices[@]}"; do
    if [[ "$choice" == "mnist" || "$choice" == "cifar10" ]]; then
        gpu=0
    else
        gpu=1
    fi
    
    python3 train_and_test.py --type "$choice" --gpu "$gpu"
done
