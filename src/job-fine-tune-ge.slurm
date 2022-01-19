#! /bin/sh
#SBATCH --job-name=job-fine-tune-GE
#SBATCH --output=job-fine-tune-GE.out
#SBATCH --error=job-fine-tune-GE.err
#SBATCH --partition=studentbatch
#SBATCH --time=300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

python3 run_goemotions.py --taxonomy original
