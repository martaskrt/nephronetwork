#!/bin/bash

module load python/2.7.16

cab_dir=$1

echo "Cab directory" $cab_dir

ls -lt $cab_dir/*cab | awk '{print $9}' > $cab_dir/cab-list.txt

echo "Cab list created"

while read cab_file ; do
	
	dcm_file="${cab_file%.cab}"
	
	/hpf/largeprojects/agoldenb/lauren/software/cabextract-1.9.1/cabextract -d $dcm_file $cab_file	
	
	python ./preprocess-us-seq.py -dcm_dir $dcm_file -rootdir $cab_dir
	
	rm -r $dcm_file
	
done < $cab_dir/cab-list.txt

