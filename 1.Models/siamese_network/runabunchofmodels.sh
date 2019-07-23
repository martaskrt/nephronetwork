var2=".txt"

###########################################################
###################### UNet model #########################
###########################################################
#mkdir mountpoint/20190601_CVModels
#mkdir mountpoint/20190601_fullModels

var1="/storage/20190715_CVModels/"
var3="/storage/20190715_fullModels/"

mkdir $var1
mkdir $var3

#dir_name="20190711_vanilla_siamese_female"
#python3 train_siamese_network_CV_v2.py --unet --sc 0 --init "none" --contrast 2 --batch_size 128 --epochs 100 --view 'siamese' --output_dim 256 --gender female --dir "$var1$dir_name" > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="20190711_vanilla_siamese_female_full"
#python3 train_siamese_network.py --unet --sc 0 --init "none" --contrast 2 --batch_size 128 --epochs 100 --view 'siamese' --output_dim 256 --gender female --dir "$var1$dir_name" > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="20190711_vanilla_siamese_male"
#python3 train_siamese_network_CV_v2.py --unet --sc 0 --init "none" --contrast 2 --batch_size 128 --epochs 35 --view 'siamese' --output_dim 256 --gender male --dir "$var1$dir_name" > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="20190711_vanilla_siamese_male_full"
#python3 train_siamese_network.py --unet --sc 0 --init "none" --contrast 2 --batch_size 128 --epochs 35 --view 'siamese' --output_dim 256 --gender male --dir "$var1$dir_name" > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"


#dir_name="20190711_vanilla_sag_female"
#python3 train_siamese_network_CV_v2.py --unet --sc 0 --init "none" --contrast 2 --batch_size 128 --epochs 50 --view 'sag' --output_dim 256 --gender female --dir "$var1$dir_name" > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="20190711_vanilla_sag_female_full"
#python3 train_siamese_network.py --unet --sc 0 --init "none" --contrast 2 --batch_size 128 --epochs 50 --view 'sag' --output_dim 256 --gender female --dir "$var1$dir_name" > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="20190711_vanilla_sag_male"
#python3 train_siamese_network_CV_v2.py --unet --sc 0 --init "none" --contrast 2 --batch_size 128 --epochs 35 --view 'sag' --output_dim 256 --gender male --dir "$var1$dir_name" > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="20190711_vanilla_sag_male_full"
#python3 train_siamese_network.py --unet --sc 0 --init "none" --contrast 2 --batch_size 128 --epochs 35 --view 'sag' --output_dim 256 --gender male --dir "$var1$dir_name" > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

dir_name="20190715_vanilla_siamese_hydroonly"
python3 train_siamese_network_CV_v2.py --hydro_only --unet --sc 0 --init "none" --contrast 2 --batch_size 128 --epochs 35 --view 'siamese' --output_dim 256 --dir "$var1$dir_name" > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"


dir_name="20190715_vanilla_siamese_hydroonly_full"
python3 train_siamese_network.py --hydro_only --unet --sc 0 --init "none" --contrast 2 --output_dim 256 --batch_size 128  --dir "$var3$dir_name" --epochs 35  --view 'siamese' > "$dir_name$var2" && mv "$dir_name$var2" "$var3$dir_name"

#dir_name="20190703_vanilla_siamese_hydroonly4realz_fullModelPretraining_full"
#python3 train_siamese_network.py --hydro_only --unet --sc 0 --init "none" --contrast 2 --output_dim 256 --batch_size 128  --dir "$var3$dir_name" --epochs 35  --view 'siamese'  --checkpoint /storage/20190619_fullModels/20190619_vanilla_siamese_eval_dropout1_0.5_full/checkpoint_21.pth > "$dir_name$var2" && mv "$dir_name$var2" "$var3$dir_name"


#dir_name="20190619_vanilla_trans_dropout1_0.5_128_2_hydroonly"
#python3 train_siamese_network_CV_v2.py --hydro_only --unet --sc 0 --init "none" --contrast 2 --batch_size 128 --epochs 35 --view 'trans' --output_dim 128 --dir "$var1$dir_name" > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="20190619_vanilla_trans_dropout1_0.5_128_2_hydroonly_full"
#python3 train_siamese_network.py --hydro_only --unet --sc 0 --init "none" --contrast 2 --output_dim 128 --batch_size 128  --dir "$var3$dir_name" --epochs 35  --view 'trans'  > "$dir_name$var2" && mv "$dir_name$var2" "$var3$dir_name"

#dir_name="20190619_vanilla_sag_dropout1_0.5_128_2_hydroonly"
#python3 train_siamese_network_CV_v2.py --hydro_only --unet --sc 0 --init "none" --contrast 2 --batch_size 128 --epochs 35 --view 'sag' --output_dim 128 --dir "$var1$dir_name" > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="20190619_vanilla_sag_dropout1_0.5_128_2_hydroonly_full"
#python3 train_siamese_network.py --hydro_only --unet --sc 0 --init "none" --contrast 2 --output_dim 128 --batch_size 128  --dir "$var3$dir_name" --epochs 35  --view 'sag'  > "$dir_name$var2" && mv "$dir_name$var2" "$var3$dir_name"

#dir_name="20190612_vanilla_256_trans"
#python3 train_siamese_network_CV_v2.py --unet --sc 0 --contrast 2 --batch_size 128 --epochs 35 --view 'trans' --output_dim 256 --dir "$var1$dir_name" > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="20190612_vanilla_256_trans_full"
#python3 train_siamese_network.py --unet --sc 0 --contrast 2 --output_dim 256 --batch_size 128  --dir "$var3$dir_name" --epochs 35  --view 'trans'  > "$dir_name$var2" && mv "$dir_name$var2" "$var3$dir_name"

#dir_name="20190612_vanilla_sag_128"
#python3 train_siamese_network_CV_v2.py --unet --sc 0 --init "none" --contrast 2 --batch_size 128 --epochs 35 --view 'sag' --output_dim 128 --dir "$var1$dir_name" > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="20190612_vanilla_sag_128_full"
#python3 train_siamese_network.py --unet --sc 0 --init "none" --contrast 2 --output_dim 128 --batch_size 128  --dir "$var3$dir_name" --epochs 35  --view 'sag'  > "$dir_name$var2" && mv "$dir_name$var2" "$var3$dir_name"

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
