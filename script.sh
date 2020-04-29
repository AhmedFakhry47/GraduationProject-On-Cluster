#!/bin/sh
#SBATCH --account=g.alex054
#SBATCH --job-name=gpu_devices
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --priority=TOP
#SBATCH --no-requeue
#SBATCH --ntasks=1
#SBATCH --time=32:00:00
#SBATCH --output=nooccams.out
#SBATCH --mail-user=ahmedfakhry805@gmail.com
#SBATCH --mail-type=ALL,TIME_LIMIT_10


python -u nutshell/nooccams.py


