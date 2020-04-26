#/bin/bash

python3 /home/marta/nephronetwork-github/nephronetwork/DMSA/Models/view_lab/train_view_func.py \
	-func_train_datasheet /home/marta/nephronetwork-github/nephronetwork/1.Models/siamese_network/mountpoint/data/load_training_test_sets/DMSA-train-datasheet-top3view-USfunc-noVlab.csv \
	-func_val_datasheet /home/marta/nephronetwork-github/nephronetwork/1.Models/siamese_network/mountpoint/data/load_training_test_sets/DMSA-val-datasheet-top3view-USfunc-noVlab.csv \
	-func_test_datasheet /home/marta/nephronetwork-github/nephronetwork/1.Models/siamese_network/mountpoint/data/load_training_test_sets/DMSA-test-datasheet-top3view-USfunc-noVlab.csv \
	-lab_train_datasheet /home/marta/nephronetwork-github/nephronetwork/1.Models/siamese_network/mountpoint/data/load_training_test_sets/train-view_label_df_20200423-noOther.csv \
	-lab_val_datasheet /home/marta/nephronetwork-github/nephronetwork/1.Models/siamese_network/mountpoint/data/load_training_test_sets/val-view_label_df_20200423-noOther.csv \
	-lab_test_datasheet /home/marta/nephronetwork-github/nephronetwork/1.Models/siamese_network/mountpoint/data/load_training_test_sets/test-view_label_df_20200423-noOther.csv \
	-csv_outdir /home/marta/lauren \
	-us_dir /home/marta/nephronetwork-github/nephronetwork/1.Models/siamese_network/mountpoint/all-jpgs-dmsa/ \
	-lab_us_dir /home/marta/nephronetwork-github/nephronetwork/1.Models/siamese_network/mountpoint/label_img/ \
	-lr 0.001 \
	-max_epochs 9 \
	-dichot	\
	-RL	
	
