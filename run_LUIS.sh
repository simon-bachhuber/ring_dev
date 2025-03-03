#!/bin/bash -l
#SBATCH --gres=gpu:l40s:4
#SBATCH --time=200:00:00
#SBATCH --partition=imes.gpu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=10G

export ON_CLUSTER=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=offline

# Load modules
module load Miniforge3

# Activate Env
conda activate ring

# STEP 1: Generate data
srun python ring_dev/train_step1_generateData_v2.py ...

# STEP 2: Training Network
srun python ring_dev/train_step2_trainRing_v2.py ...