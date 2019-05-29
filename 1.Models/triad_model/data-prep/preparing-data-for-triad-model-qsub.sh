#!/bin/bash

#PBS -l walltime=23:00:00,mem=20g,vmem=20g

module load python/2.7.16

python /hpf/largeprojects/agoldenb/lauren/Hydronephrosis/triad_model/code/preparing-data-for-triad-model_ind_pickles.py
