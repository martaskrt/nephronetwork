#!/bin/bash
#PBS -q gpu -l nodes=1:ppn=1:gpus=1,vmem=40g,mem=40g,walltime=06:00:00

module load anaconda
module load glibc
source activate nnet

python3 /home/delvinso/nephro/nephro_net/predict_bladder.py
