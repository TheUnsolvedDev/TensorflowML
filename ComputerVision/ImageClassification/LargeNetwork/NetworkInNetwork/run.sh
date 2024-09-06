choices=("mnist" "cifar10" "cifar100" "fashion_mnist")
for choice in "${choices[@]}"; do
    python3 train_and_test.py --type $choice --gpu -1
done