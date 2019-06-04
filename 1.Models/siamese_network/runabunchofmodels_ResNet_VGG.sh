#!/bin/bash

declare -a my_lrs=("0.001"  "0.005")
declare -a my_views=("trans" "sag" "siamese")





var2=".txt"

###########################################################
###################### UNet model #########################
###########################################################
mkdir mountpoint/20190601_CVModels
mkdir mountpoint/20190601_fullModels

var1="mountpoint/20190601_CVModels"
var3="mountpoint/20190601_fullModels"

dir_name="unet_20190601_vanilla_siamese"
python3 train_siamese_network_CV_v2.py --dir "$var1$dir_name" --epochs 35  --view 'siamese'  --unet --output_dim 256 > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

dir_name="unet_20190601_vanilla_siamese_full"
python3 train_siamese_network.py --output_dim 256 --dir "$var3$dir_name" --epochs 35  --view 'siamese' --unet  > "$dir_name$var2" && mv "$dir_name$var2" "$var3$dir_name"

dir_name="unet_20190601_vanilla_sag"
python3 train_siamese_network_CV_v2.py --dir "$var1$dir_name" --epochs 35  --view 'sag'  --unet --output_dim 128 > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

dir_name="unet_20190601_vanilla_sag_full"
python3 train_siamese_network.py --output_dim 128 --dir "$var3$dir_name" --epochs 35  --view 'sag' --unet  > "$dir_name$var2" && mv "$dir_name$var2" "$var3$dir_name"

dir_name="unet_20190601_vanilla_trans"
python3 train_siamese_network_CV_v2.py --dir "$var1$dir_name" --epochs 35  --view 'trans'  --unet --output_dim 128 > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

dir_name="unet_20190601_vanilla_trans_full"
python3 train_siamese_network.py --output_dim 128 --dir "$var3$dir_name" --epochs 35  --view 'trans' --unet  > "$dir_name$var2" && mv "$dir_name$var2" "$var3$dir_name"

#dir_name="unet_20190524_vanilla_siamese_256_2"
#python3 train_siamese_network_CV_v2.py --output_dim 256 --dir "$var1$dir_name" --epochs 25  --view 'siamese'  --unet > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="siamnet_20190524_vanilla_sag_256_2"
#python3 train_siamese_network_CV_v2.py --output_dim 256 --dir "$var1$dir_name" --epochs 25  --view 'sag' > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="unet_20190524_vanilla_sag_256_2"
#python3 train_siamese_network_CV_v2.py --output_dim 256 --dir "$var1$dir_name" --epochs 25  --view 'sag' --unet > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"


#dir_name="siamnet_20190524_vanilla_trans_128_2_f"
#python3 train_siamese_network_CV_v2.py --output_dim 128 --dir "$var1$dir_name" --epochs 35  --view 'trans' > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="unet_20190524_vanilla_trans_256_2"
#python3 train_siamese_network_CV_v2.py --output_dim 256 --dir "$var1$dir_name" --epochs 25  --view 'trans' --unet > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="unet_20190524_vanilla_sag_256"
#python3 train_siamese_network_CV_v2.py --output_dim 256 --dir "$var1$dir_name" --epochs 25  --view 'sag'  --unet > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="unet_20190524_vanilla_trans_512"
#python3 train_siamese_network_CV_v2.py --output_dim 512 --dir "$var1$dir_name" --epochs 25  --view 'trans'  --unet > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="siamnet_20190524_vanilla_siamese_256_35e"
#python3 train_siamese_network_CV_v2.py --output_dim 256 --dir "$var1$dir_name" --epochs 35  --view 'siamese' > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="siamnet_20190524_vanilla_sag_512"
#python3 train_siamese_network_CV_v2.py --output_dim 512 --dir "$var1$dir_name" --epochs 25  --view 'sag' > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="unet_20190522_vanilla_siamese_full_128_REDO2"
#python3 train_siamese_network.py --output_dim 128 --dir "$var1$dir_name" --epochs 25  --view 'siamese' --unet  > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="siamnet_20190524_vanilla_trans_512"
#python3 train_siamese_network_CV_v2.py --output_dim 512 --dir "$var1$dir_name" --epochs 25  --view 'trans' > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

###########################################################
################## 512 full model #########################
###########################################################

#var1="mountpoint/20190524_models_fullTrain/"

#dir_name="unet_20190522_vanilla_siamese_full_128_REDO2"
#python3 train_siamese_network.py --output_dim 128 --dir "$var1$dir_name" --epochs 25  --view 'siamese' --unet  > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="unet_20190522_vanilla_siamese_full_256_REDO2"
#python3 train_siamese_network.py --output_dim 256 --dir "$var1$dir_name" --epochs 20  --view 'siamese' --unet  > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="unet_20190522_vanilla_siamese_full_512_REDO2"
#python3 train_siamese_network.py --output_dim 512 --dir "$var1$dir_name" --epochs 20  --view 'siamese' --unet  > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="unet_20190522_vanilla_siamese_full_1024_REDO2"
#python3 train_siamese_network.py --output_dim 1024 --dir "$var1$dir_name" --epochs 20  --view 'siamese' --unet  > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="unet_20190522_vanilla_siamese_full_4096_REDO2"
#python3 train_siamese_network.py --output_dim 4096 --dir "$var1$dir_name" --epochs 20  --view 'siamese' --unet  > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="unet_20190522_vanilla_sag_full_512"
#python3 train_siamese_network.py --output_dim 512 --dir "$var1$dir_name" --epochs 25  --view 'sag' --unet  > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="unet_20190522_vanilla_trans_full_512"
#python3 train_siamese_network.py --output_dim 512 --dir "$var1$dir_name" --epochs 25  --view 'trans' --unet  > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="siamnet_20190522_vanilla_siamese_full_256_REDO"
#python3 train_siamese_network.py --output_dim 256 --dir "$var1$dir_name" --epochs 25  --view 'siamese'  > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="siamnet_20190522_vanilla_sag_full_128_REDO"
#python3 train_siamese_network.py --output_dim 128 --dir "$var1$dir_name" --epochs 20  --view 'sag'  > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="siamnet_20190522_vanilla_trans_full_128_REDO"
#python3 train_siamese_network.py --output_dim 128 --dir "$var1$dir_name" --epochs 20  --view 'trans'  > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="siamnet_20190522_vanilla_trans_full_512"
#python3 train_siamese_network.py --output_dim 512 --dir "$var1$dir_name" --epochs 25  --view 'trans'  > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

