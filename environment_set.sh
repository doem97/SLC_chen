#!/bin/bash

# upload current file to the cloud to use, you should modify by yourself first.

# for tensorflow Docker (vast.ai):
# apt update && apt install -y git-core wget libsm6 libxext6 libxrender-dev
# pip install keras opencv-python pandas tqdm

CURRENT=`pwd`
DATA_PATH="${CURRENT}/SLC_chen/dataset"
IMAGE_ID="resized_128_128"

git clone https://github.com/doem97/SLC_chen.git
# will init a ${IMAGE_ID} folder for quick-experiment.
wget -P $DATA_PATH https://storage.googleapis.com/skin-lesion-classification_bucket/${IMAGE_ID}.zip
unzip -q "${DATA_PATH}/${IMAGE_ID}.zip" -d $DATA_PATH
rm "${DATA_PATH}/${IMAGE_ID}.zip"
echo "${IMAGE_ID} downloaded finished and can be train."