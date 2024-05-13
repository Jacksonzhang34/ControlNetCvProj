#!/bin/bash

source /etc/profile
module load anaconda/2023a-pytorch

prompts=("prompt1_semantic_centroid" "prompt2_cat4")

echo "Running job $LLSUB_RANK \n\n"

python3 model_training.py --prompt "${prompts[$LLSUB_RANK]}"
