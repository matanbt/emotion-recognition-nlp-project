#! /bin/sh
#SBATCH --job-name=job-fine-tune-ge
#SBATCH --output=results/job-fine-tune-ge.out
#SBATCH --error=results/job-fine-tune-ge.err
#SBATCH --partition=studentbatch
#SBATCH --time=300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

python3 src/main.py --config regression.json
