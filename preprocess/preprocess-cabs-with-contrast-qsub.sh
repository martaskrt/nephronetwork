#!/bin/bash

#PBS -l walltime=23:00:00,mem=40g,vmem=40g

module load python/2.7.16

cab_dir=/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/all-cabs/

echo "Cab directory" $cab_dir

ls -lt $cab_dir/*cab | awk '{print $9}' > $cab_dir/cab-list.txt

echo "Cab list created"

while read cab_file ; do
	
	dcm_file="${cab_file%.cab}"
	
	/hpf/largeprojects/agoldenb/lauren/software/cabextract-1.9.1/cabextract -d $dcm_file $cab_file	
	
	python /hpf/largeprojects/agoldenb/lauren/Hydronephrosis/data/load_training_test_sets/preprocess-us-seq-with-contrast.py -dcm_dir $dcm_file -rootdir $cab_dir -contrast 2 -jpg_dump_dir /hpf/largeprojects/agoldenb/lauren/Hydronephrosis/train-jpgs/contrast2/ -out_dir /hpf/largeprojects/agoldenb/lauren/Hydronephrosis/train-us-seqs/contrast2/
	
	rm -r $dcm_file
	
done < $cab_dir/cab-list.txt

