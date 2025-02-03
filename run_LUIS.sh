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
srun python ring_dev/train_step1_generateData_v2.py 82944 $BIGWORK/data/v2_lam4_kin_rigid_rom_pm

# STEP 2: Training Network
srun python ring_dev/train_step2_trainRing_v2.py $BIGWORK/data/v2_lam4_kin_rigid_rom_pm 256 10000 --use-wandb --wandb-name "RING-Massive-nonSparse" --exp-cbs --lr 3e-4 --tbp 150 --rnn-w 800 --rnn-d 8 --lin-w 200 --lin-d 0 --n-val 256 --seed 58 --drop-dof 0.5 --drop-ja-1d 1 --drop-ja-2d 1 --rand-ori --drop-imu-1d 0.0 --drop-imu-2d 0.0 --drop-imu-3d 0.0 --celltype "gru"