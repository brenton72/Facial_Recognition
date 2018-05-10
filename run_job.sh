#!/bin/bash
#
#BATCH --job-name=facial
#SBATCH --gres=gpu:1
#SBATCH --time=47:00:00
#SBATCH --mem=15GB
#SBATCH --output=outputs/%A.out
#SBATCH --error=outputs/%A.err

module purge
module load python3/intel/3.5.3 pytorch/python3.5/0.2.0_3 torchvision/python3.5/0.1.9 
python3 -m pip install opencv-python --user


cd /scratch/sb3923/Facial_Recognition

python3 -u train_resnet.py $1 --experiment $2 > logs/$2.log
