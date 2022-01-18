#! /bin/sh
#SBATCH --job-name=myjob
#SBATCH --output=first-train-goemotions.out
#SBATCH --error=first-train-goemotions.err
#SBATCH --partition=studentbatch
#SBATCH --time=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

python3 run_goemotions.py --taxonomy original
