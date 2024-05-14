#!/bin/bash

WEIGHT_ID=0

source /etc/profile
module load anaconda/2023a-pytorch

weights=("prompt1_random" "prompt1_gpt_best" "prompt2_cat2" "prompt2_cat5" "prompt2_sum" "prompt1_longest" "prompt1_tf_idf" "prompt1_semantic_centroid" "prompt2_cat3" "prompt2_cat4" "prompt1_random_1" "prompt1_longest_1")

prompts=("prompt1_random" "prompt1_gpt_best" "prompt2_cat2" "prompt2_cat5" "prompt2_sum" "prompt1_longest" "prompt1_tf_idf" "prompt1_semantic_centroid" "prompt2_cat3" "prompt2_cat4" "prompt1_random" "prompt1_longest")

trained=("true" "false")

echo "Running weights $WEIGHT_ID task $LLSUB_RANK with trained ${trained[$LLSUB_RANK]}"

python3 evaluation.py --weightName "${weights[$WEIGHT_ID]}" --prompt "${prompts[$WEIGHT_ID]}" --trained "${trained[$LLSUB_RANK]}"
