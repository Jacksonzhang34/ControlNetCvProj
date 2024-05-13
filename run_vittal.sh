#!/bin/bash

source /etc/profile
module load anaconda/2023a-pytorch

prompts=("prompt1_random" "prompt1_longest" "prompt1_tf_idf" "prompt1_gpt_best" "prompt2_cat2" "prompt2_cat3" "prompt2_cat5" "prompt2_sum" "prompt1_semantic_centroid" "prompt2_cat4")

echo "Running job $LLSUB_RANK \n\n"

python3 model_training.py --prompt "${prompts[$LLSUB_RANK]}"