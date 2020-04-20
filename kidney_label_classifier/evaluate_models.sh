#!/bin/bash
#PBS -q gpu -l nodes=1:ppn=1:gpus=1,vmem=20g,mem=20g,walltime=04:00:00

module load anaconda
module load glibc
source activate nnet

python3 /home/delvinso/nephro/nephro_net/evaluate_models.py


