#!/bin/bash
#SBATCH --job-name=tune_4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=4-00:00:00

#SBATCH --output=/home/pravindra.cse22.itbhu/projects/logs/4_output.log
#SBATCH --error=/home/pravindra.cse22.itbhu/projects/logs/4_error.log

module load cuda/12.6.3
module load python3.8/3.8

cd /home/pravindra.cse22.itbhu/projects/
source myenv/bin/activate

python3 -u cv_fold_4.py

exit 0
