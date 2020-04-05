#!/bin/sh
#SBATCH --account=g.alex054
#SBATCH --job-name=plotting
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:00:02
#SBATCH --output=plot.out


python -u nutshell/cleaner.py occamsout.txt figure.png
