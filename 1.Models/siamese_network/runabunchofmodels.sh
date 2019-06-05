var2=".txt"

###########################################################
###################### UNet model #########################
###########################################################
#mkdir mountpoint/20190601_CVModels
#mkdir mountpoint/20190601_fullModels

var1="mountpoint/20190601_CVModels/"
var3="mountpoint/20190601_fullModels/"



#dir_name="siamnet_20190601_vanilla_siamese_gap"
#python3 train_siamese_network_CV_v2.py --dir "$var1$dir_name" --epochs 35  --view 'siamese'  --output_dim 256 > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="siamnet_20190601_vanilla_siamese_gap_full"
#python3 train_siamese_network.py --output_dim 256 --dir "$var3$dir_name" --epochs 35  --view 'siamese' > "$dir_name$var2" && mv "$dir_name$var2" "$var3$dir_name"

#dir_name="siamnet_20190601_vanilla_siamese"
#python3 train_siamese_network_CV_v2.py --dir "$var1$dir_name" --epochs 35  --view 'siamese'  --output_dim 256 > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="siamnet_20190601_vanilla_siamese_full"
#python3 train_siamese_network.py --output_dim 256 --dir "$var3$dir_name" --epochs 35  --view 'siamese' > "$dir_name$var2" && mv "$dir_name$var2" "$var3$dir_name"

dir_name="unet_20190601_vanilla_sag_2"
python3 train_siamese_network_CV_v2.py --unet --dir "$var1$dir_name" --epochs 35  --view 'sag'  --output_dim 128 > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

dir_name="unet_20190601_vanilla_sag_full_2"
python3 train_siamese_network.py --unet --output_dim 128 --dir "$var3$dir_name" --epochs 35  --view 'sag'  > "$dir_name$var2" && mv "$dir_name$var2" "$var3$dir_name"

#dir_name="siamnet_20190601_vanilla_trans"
#python3 train_siamese_network_CV_v2.py --dir "$var1$dir_name" --epochs 35  --view 'trans' --output_dim 128 > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="siamnet_20190601_vanilla_trans_full"
#python3 train_siamese_network.py --output_dim 128 --dir "$var3$dir_name" --epochs 35  --view 'trans'  > "$dir_name$var2" && mv "$dir_name$var2" "$var3$dir_name"

#dir_name="unet_20190601_vanilla_siamese_gap"
#python3 train_siamese_network_CV_v2.py --unet --dir "$var1$dir_name" --epochs 35  --view 'siamese'  --output_dim 256 > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="unet_20190601_vanilla_siamese_gap_full"
#python3 train_siamese_network.py --unet --output_dim 256 --dir "$var3$dir_name" --epochs 35  --view 'siamese' > "$dir_name$var2" && mv "$dir_name$var2" "$var3$dir_name"
