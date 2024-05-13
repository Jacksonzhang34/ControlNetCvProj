#!/bin/bash
# SBATCH -N 1
# SBATCH -a 0-5
# SBATCH --job-name train_vlm
# SBATCH --gres=gpu:volta:1
# SBATCH --tasks-per-node=2
# SBATCH --cpus-per-task=20

if [ ! -e /proc/$(pidof nvidia-smi) ]
then
	echo "nvidia-smi does not seem to be running. exiting job"
    exit 1
fi

source /etc/profile
module load anaconda/2023a-pytorch

export TOTAL_GPUS=${SLURM_NTASKS}
export GPUS_PER_NODE=1

export BACKEND="pytorch"

prompts=("prompt1_random" "prompt1_longest" "prompt1_tf_idf" "prompt1_semantic_centroid" "prompt1_gpt_best" "prompt2_cat2" "prompt2_cat3" "prompt2_cat4" "prompt2_cat5" "prompt2_sum")

echo ""
echo "--------------------------------------------------------------------------------------------------"
echo ""
echo "Running SLURM job"
echo ""
echo "--------------------------------------------------------------------------------------------------"
echo ""

python3 model_training.py --prompt ${prompts[$SLURM_ARRAY_TASK_ID]} 


# echo ""
# echo "=> end of script"
# echo "creating summary"
# # run some code here to summarize results

# WALL_TIME=$(sacct --format="ElapsedRaw" -j ${SLURM_JOBID} -n | head -n1 | awk '{$1=$1};1')
# echo "\"WALL_TIME\" : \"${WALL_TIME}\""
# echo "}"

# echo "all done"
