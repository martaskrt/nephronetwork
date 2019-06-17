var2=".txt"

###########################################################
###################### UNet model #########################
###########################################################
#mkdir mountpoint/20190601_CVModels
#mkdir mountpoint/20190601_fullModels

var1="/storage/20190611_CVModels/"
var3="/storage/20190611_fullModels/"

mkdir $var1
mkdir $var3


dir_name="unet_20190612_vanilla_siamese_upconv_1c_1ch_noinit_bs256"
python3 train_siamese_network_CV_v2.py --unet --sc 0 --init "none" --contrast 2 --batch_size 256 --epochs 35 --view 'siamese' --output_dim 256 --dir "$var1$dir_name" > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

dir_name="unet_20190612_vanilla_siamese_upconv_1c_1ch_noinit_bs256_full"
python3 train_siamese_network.py --unet --sc 0 --init "none" --contrast 2 --output_dim 256 --batch_size 256  --dir "$var3$dir_name" --epochs 35  --view 'siamese'  > "$dir_name$var2" && mv "$dir_name$var2" "$var3$dir_name"
dir_name="unet_20190611_vanilla_siamese_upconv_1c_1ch_contrast2_noLRN"
python3 train_siamese_network_CV_v2.py --unet --sc 0 --upconv 1 --contrast 2 --batch_size 128 --epochs 35 --view 'siamese' --output_dim 256 --dir "$var1$dir_name" > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

dir_name="unet_20190611_vanilla_siamese_upconv_1c_1ch_contrast2_noLRN_full"
python3 train_siamese_network.py --unet --sc 0 --upconv 1 --contrast 2 --output_dim 256 --batch_size 128  --dir "$var3$dir_name" --epochs 35  --view 'siamese'  > "$dir_name$var2" && mv "$dir_name$var2" "$var3$dir_name"


#dir_name="unet_20190606_vanilla_siamese_sc2"
#python3 train_siamese_network_CV_v2.py --unet --sc 2 --batch_size 128 --epochs 35 --view 'siamese' --output_dim 256 --dir "$var1$dir_name" > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="unet_20190606_vanilla_siamese_sc2_full"
#python3 train_siamese_network.py --unet --sc 2 --output_dim 256 --batch_size 128  --dir "$var3$dir_name" --epochs 35  --view 'siamese'  > "$dir_name$var2" && mv "$dir_name$var2" "$var3$dir_name"

#dir_name="unet_20190606_vanilla_siamese_sc1"
#python3 train_siamese_network_CV_v2.py --unet --sc 1 --batch_size 128 --epochs 35 --view 'siamese' --output_dim 256 --dir "$var1$dir_name" > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="unet_20190606_vanilla_siamese_sc1_full"
#python3 train_siamese_network.py --unet --sc 1 --output_dim 256 --batch_size 128  --dir "$var3$dir_name" --epochs 35  --view 'siamese'  > "$dir_name$var2" && mv "$dir_name$var2" "$var3$dir_name"

#dir_name="unet_20190606_vanilla_siamese_upconv"
#python3 train_siamese_network_CV_v2.py --unet --sc 0 --batch_size 128 --epochs 35 --view 'siamese' --output_dim 256 --dir "$var1$dir_name" > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="unet_20190606_vanilla_siamese_upconv_full"
#python3 train_siamese_network.py --unet --sc 0 --output_dim 256 --batch_size 128  --dir "$var3$dir_name" --epochs 35  --view 'siamese'  > "$dir_name$var2" && mv "$dir_name$var2" "$var3$dir_name"
#dir_name="siamnet_20190601_vanilla_siamese_gap"
#python3 train_siamese_network_CV_v2.py --dir "$var1$dir_name" --epochs 35  --view 'siamese'  --output_dim 256 > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="siamnet_20190601_vanilla_siamese_gap_full"
#python3 train_siamese_network.py --output_dim 256 --dir "$var3$dir_name" --epochs 35  --view 'siamese' > "$dir_name$var2" && mv "$dir_name$var2" "$var3$dir_name"

#dir_name="siamnet_20190601_vanilla_siamese"
#python3 train_siamese_network_CV_v2.py --dir "$var1$dir_name" --epochs 35  --view 'siamese'  --output_dim 256 > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="siamnet_20190601_vanilla_siamese_full"
#python3 train_siamese_network.py --output_dim 256 --dir "$var3$dir_name" --epochs 35  --view 'siamese' > "$dir_name$var2" && mv "$dir_name$var2" "$var3$dir_name"

#dir_name="unet_20190601_vanilla_sag_2"
#python3 train_siamese_network_CV_v2.py --unet --dir "$var1$dir_name" --epochs 35  --view 'sag'  --output_dim 128 > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="unet_20190601_vanilla_sag_full_2"
#python3 train_siamese_network.py --unet --output_dim 128 --dir "$var3$dir_name" --epochs 35  --view 'sag'  > "$dir_name$var2" && mv "$dir_name$var2" "$var3$dir_name"

#dir_name="siamnet_20190601_vanilla_trans"
#python3 train_siamese_network_CV_v2.py --dir "$var1$dir_name" --epochs 35  --view 'trans' --output_dim 128 > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="siamnet_20190601_vanilla_trans_full"
#python3 train_siamese_network.py --output_dim 128 --dir "$var3$dir_name" --epochs 35  --view 'trans'  > "$dir_name$var2" && mv "$dir_name$var2" "$var3$dir_name"

#dir_name="unet_20190601_vanilla_siamese_gap"
#python3 train_siamese_network_CV_v2.py --unet --dir "$var1$dir_name" --epochs 35  --view 'siamese'  --output_dim 256 > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="unet_20190601_vanilla_siamese_gap_full"
#python3 train_siamese_network.py --unet --output_dim 256 --dir "$var3$dir_name" --epochs 35  --view 'siamese' > "$dir_name$var2" && mv "$dir_name$var2" "$var3$dir_name"
