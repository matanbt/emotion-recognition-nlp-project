#! /bin/sh
#SBATCH --job-name=job-fine-tune-ge
#SBATCH --output=results/job-fine-tune-ge.out
#SBATCH --error=results/job-fine-tune-ge.err
#SBATCH --partition=studentbatch
#SBATCH --time=2000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

echo ">>>> 0.0"
python3 src/main.py --config exp0.0.json
echo ">>>> 0.01"
python3 src/main.py --config exp0.01.json
echo ">>>> 0.05"
python3 src/main.py --config exp0.05.json
echo ">>>> 0.10"
python3 src/main.py --config exp0.10.json
echo ">>>> 0.25"
python3 src/main.py --config exp0.25.json
echo ">>>> 0.50"
python3 src/main.py --config exp0.50.json
echo ">>>> 0.75"
python3 src/main.py --config exp0.75.json
echo ">>>> 0.90"
python3 src/main.py --config exp0.90.json
echo ">>>> 0.95"
python3 src/main.py --config exp0.95.json
echo ">>>> 0.99"
python3 src/main.py --config exp0.99.json
echo ">>>> 1.0"
python3 src/main.py --config exp1.0.json
echo "DONE"