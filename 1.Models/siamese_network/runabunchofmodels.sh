var2=".txt"

###########################################################
###################### UNet model #########################
###########################################################
#mkdir mountpoint/20190601_CVModels
#mkdir mountpoint/20190601_fullModels

var1="/storage/20190903_CVModels/"
var3="/storage/20190903_fullModels/"

mkdir $var1
mkdir $var3




#dir_name="prehdict_20190802_vanilla_siamese_dim256_c1"
#python3 train_siamese_network_CV_v2.py --unet --sc 0 --init "none" --contrast 1 --batch_size 128 --epochs 35 --view 'siamese' --output_dim 256 --dir "$var1$dir_name" > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="prehdict_20190802_vanilla_siamese_dim256_c1_full"
#python3 train_siamese_network.py --unet --sc 0 --init "none" --contrast 1 --batch_size 128 --epochs 35 --view 'siamese' --output_dim 256 --dir "$var3$dir_name" > "$dir_name$var2" && mv "$dir_name$var2" "$var3$dir_name"

#dir_name="prehdict_20190802_vanilla_siamese_dim512_c1"
#python3 train_siamese_network_CV_v2.py --unet --sc 0 --init "none" --contrast 1 --batch_size 128 --epochs 35 --view 'siamese' --output_dim 512 --dir "$var1$dir_name" > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="prehdict_20190802_vanilla_siamese_dim512_c1_full"
#python3 train_siamese_network.py --unet --sc 0 --init "none" --contrast 1 --batch_size 128 --epochs 35 --view 'siamese' --output_dim 512 --dir "$var3$dir_name" > "$dir_name$var2" && mv "$dir_name$var2" "$var3$dir_name"

#dir_name="prehdict_20190815_jigsawunet_siamese_c1"
#python3 train_siamese_network_CV_v2.py --unet --sc 0 --init "none" --contrast 1 --batch_size 128 --epochs 35 --view 'siamese' --dir "$var1$dir_name" --checkpoint ../../models/JigsawPuzzlePytorch-master/checkpoints_20190415_bs256_e70_lr0.01_64_c1/jps_069_014770.pth.tar --dir "$var1$dir_name" > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="prehdict_20190815_jigsawunet_siamese_c1_full"
#python3 train_siamese_network.py --unet --sc 0 --init "none" --contrast 1 --batch_size 128 --epochs 35 --view 'siamese' --dir "$var1$dir_name" --checkpoint ../../models/JigsawPuzzlePytorch-master/checkpoints_20190415_bs256_e70_lr0.01_64_c1/jps_069_014770.pth.tar --dir "$var3$dir_name" > "$dir_name$var2" && mv "$dir_name$var2" "$var3$dir_name"


#dir_name="prehdict_20190815_jigsawunet_sag_c1"
#python3 train_siamese_network_CV_v2.py --unet --sc 0 --init "none" --contrast 1 --batch_size 128 --epochs 35 --view 'sag' --dir "$var1$dir_name" --checkpoint ../../models/JigsawPuzzlePytorch-master/checkpoints_20190415_bs256_e70_lr0.01_64_c1/jps_069_014770.pth.tar --dir "$var1$dir_name" > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="prehdict_20190815_jigsawunet_sag_c1_full"
#python3 train_siamese_network.py --unet --sc 0 --init "none" --contrast 1 --batch_size 128 --epochs 35 --view 'sag' --dir "$var1$dir_name" --checkpoint ../../models/JigsawPuzzlePytorch-master/checkpoints_20190415_bs256_e70_lr0.01_64_c1/jps_069_014770.pth.tar --dir "$var3$dir_name" > "$dir_name$var2" && mv "$dir_name$var2" "$var3$dir_name"

#dir_name="prehdict_20190815_jigsawunet_trans_c1"
#python3 train_siamese_network_CV_v2.py --unet --sc 0 --init "none" --contrast 1 --batch_size 128 --epochs 35 --view 'trans' --dir "$var1$dir_name" --checkpoint ../../models/JigsawPuzzlePytorch-master/checkpoints_20190415_bs256_e70_lr0.01_64_c1/jps_069_014770.pth.tar --dir "$var1$dir_name" > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="prehdict_20190815_jigsawunet_trans_c1_full"
#python3 train_siamese_network.py --unet --sc 0 --init "none" --contrast 1 --batch_size 128 --epochs 35 --view 'trans' --dir "$var1$dir_name" --checkpoint ../../models/JigsawPuzzlePytorch-master/checkpoints_20190415_bs256_e70_lr0.01_64_c1/jps_069_014770.pth.tar --dir "$var3$dir_name" > "$dir_name$var2" && mv "$dir_name$var2" "$var3$dir_name"
#dir_name="prehdict_20190802_vanilla_trans_GAP_c1"
#python3 train_siamese_network_CV_v2.py --unet --sc 0 --init "none" --contrast 1 --batch_size 128 --epochs 35 --view 'trans' --output_dim 256 --dir "$var1$dir_name" > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"
dir_name="prehdict_20190903_octunet_siamese_c1"
python3 train_siamese_network_CV_v2.py --unet --sc 0 --init "none" --contrast 1 --batch_size 128 --epochs 35 --view 'siamese'  --checkpoint ../transfer_learning/carson/transfer_learning_checkpoints/checkpoint_oct_3.pth --dir "$var1$dir_name" > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

dir_name="prehdict_20190903_octunet_siamese_c1_full"
python3 train_siamese_network.py --unet --sc 0 --init "none" --contrast 1 --batch_size 128 --epochs 35 --view 'siamese'  --checkpoint ../transfer_learning/carson/transfer_learning_checkpoints/checkpoint_oct_3.pth --dir "$var3$dir_name" > "$dir_name$var2" && mv "$dir_name$var2" "$var3$dir_name"
