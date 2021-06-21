#!/bin/bash

#SBATCH --job-name=Resnet50-train-on-clip

#SBATCH --ntasks=1               # Number of tasks (see below)
#SBATCH --cpus-per-task=16        # Number of CPU cores per task
#SBATCH --mem=200G
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --output=ResNet50-clip%j.out  # File to which STDOUT will be written
#SBATCH --error=ResNet50-clip%j.err   # File to which STDERR will be written
#SBATCH --gres=gpu:4
#SBATCH --constraint=ImageNet2012   # Constrain to nodes where ImageNet is quickly available
#SBATCH --partition=gpu-2080ti

# Print info about current job
scontrol show job $SLURM_JOB_ID

singularity exec --nv --bind /scratch_local/ docker://lukasschott/ifr:v8 python3 train.py /path/to/dataset/ImageNet2012 -a resnet50 --dist-url 'tcp://127.0.0.1:1405' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --workers 8  --label-type soft_labels
