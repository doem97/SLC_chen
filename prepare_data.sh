#!/bin/bash

CURRENT=`pwd`
DATA_PATH="${CURRENT}/SLC_chen/dataset"

git clone https://github.com/doem97/SLC_chen.git
# will init a resized_128_128 folder for quick-experience.
wget -P $DATA_PATH https://storage.googleapis.com/skin-lesion-classification_bucket/resized_128_128.zip
unzip -q "${DATA_PATH}/resized_128_128.zip" -d $DATA_PATH
rm "${DATA_PATH}/resized_128_128.zip"
echo "resized_128_128 downloaded finished and can be train."