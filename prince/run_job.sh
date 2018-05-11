#!/bin/bash
#
#BATCH --job-name=facial
#SBATCH --gres=gpu:1
#SBATCH --time=47:00:00
#SBATCH --mem=15GB
#SBATCH --output=%A.out
#SBATCH --error=%A.err

module purge
module load python3/intel/3.5.3 pytorch/python3.5/0.2.0_3 torchvision/python3.5/0.1.9
python3 -m pip install opencv-python --user


#python3 -u train_resnet.py $1 --experiment $2 > logs/$2.log

learning_rate=$1
n_epochs=$2
experiment=$3
model_type=$4

python3 -u train_model.py  --USE_CUDA --learning_rate $learning_rate --n_epochs $n_epochs --experiment $experiment --model_type $model_type
