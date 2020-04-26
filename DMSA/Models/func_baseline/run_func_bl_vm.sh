#/bin/bash

python3 /home/marta/nephronetwork-github/nephronetwork/DMSA/Models/func_baseline/func_bl_model.py \
	-train_datasheet /home/marta/nephronetwork-github/nephronetwork/1.Models/siamese_network/mountpoint/data/load_training_test_sets/DMSA-train-datasheet-top3view-USfunc-10ksamp.csv \
	-val_datasheet /home/marta/nephronetwork-github/nephronetwork/1.Models/siamese_network/mountpoint/data/load_training_test_sets/DMSA-val-datasheet-top3view-USfunc-10ksamp.csv \
	-test_datasheet /home/marta/nephronetwork-github/nephronetwork/1.Models/siamese_network/mountpoint/data/load_training_test_sets/DMSA-test-datasheet-top3view-USfunc-10ksamp.csv \
	-csv_outdir /home/marta/lauren \
	-us_dir /home/marta/nephronetwork-github/nephronetwork/1.Models/siamese_network/mountpoint/all-jpgs-dmsa/ \
	-dmsa_dir /home/marta/nephronetwork-github/nephronetwork/1.Models/siamese_network/mountpoint/all-dmsa-cabs/dmsa-jpgs/
	-dichot
	
	
