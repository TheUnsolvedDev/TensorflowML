#!/bin/bash

FILE=$1

if [[ $FILE != "monet2photo" && $FILE != "cezanne2photo" && $FILE != "ukiyoe2photo" && $FILE != "vangogh2photo" ]]; then
    echo "Available datasets are: monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo"
    exit 1
fi

URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$FILE.zip
ZIP_FILE=Datasets/$FILE.zip
TARGET_DIR=Datasets/$FILE

wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./Datasets
rm $ZIP_FILE

# Adapt to project expected directory heriarchy
mkdir -p "$TARGET_DIR/train" "$TARGET_DIR/test"
mv "$TARGET_DIR/trainA" "$TARGET_DIR/train/A"
mv "$TARGET_DIR/trainB" "$TARGET_DIR/train/B"
mv "$TARGET_DIR/testA" "$TARGET_DIR/test/A"
mv "$TARGET_DIR/testB" "$TARGET_DIR/test/B"
