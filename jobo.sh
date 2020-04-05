#!/bin/sh
#SBATCH --account=g.alex054
#SBATCH --job-name=gpu_devices
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-4
#SBATCH --time=32:00:00
#SBATCH --output=occamsout%a.out
#SBATCH --mail-user=ahmedfakhry805@gmail.com
#SBATCH --mail-type=ALL,TIME_LIMIT_10


python -u nutshell/occamscode${SLURM_ARRAY_TASK_ID}.py
