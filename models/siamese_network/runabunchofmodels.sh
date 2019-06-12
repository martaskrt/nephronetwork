var2=".txt"

<<<<<<< Updated upstream
=======
###########################################################
#################### 512 CV model #########################
###########################################################
>>>>>>> Stashed changes

var1="/storage/20190524_models_updatedRefPts/"

#dir_name="unet_20190524_vanilla_siamese_128xtra"
#python3 train_siamese_network_CV_v2.py --output_dim 128 --dir "$var1$dir_name" --epochs 25  --view 'siamese'  --unet > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

<<<<<<< Updated upstream
dir_name="unet_20190520_vanilla_sag"
python3 train_siamese_network_CV_v2.py --dir "$var1$dir_name" --epochs 35  --view 'sag' --unet > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

dir_name="unet_20190520_vanilla_trans"
python3 train_siamese_network_CV_v2.py --dir "$var1$dir_name" --epochs 35  --view 'trans' --unet > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"
=======
#dir_name="unet_20190524_vanilla_siamese_128_2"
#python3 train_siamese_network_CV_v2.py --output_dim 128 --dir "$var1$dir_name" --epochs 25  --view 'siamese'  --unet > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="siamnet_20190524_vanilla_siamese_128_2"
#python3 train_siamese_network_CV_v2.py --output_dim 128 --dir "$var1$dir_name" --epochs 25  --view 'siamese' > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="unet_20190524_vanilla_trans_512"
#python3 train_siamese_network_CV_v2.py --output_dim 512 --dir "$var1$dir_name" --epochs 25  --view 'trans'  --unet > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="siamnet_20190524_vanilla_siamese_256_2"
#python3 train_siamese_network_CV_v2.py --output_dim 256 --dir "$var1$dir_name" --epochs 35  --view 'siamese' > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"


#dir_name="siamnet_20190524_vanilla_sag_128_2"
#python3 train_siamese_network_CV_v2.py --output_dim 128 --dir "$var1$dir_name" --epochs 25  --view 'sag' > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="siamnet_20190524_vanilla_trans_128_2"
#python3 train_siamese_network_CV_v2.py --output_dim 128 --dir "$var1$dir_name" --epochs 25  --view 'trans' > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="unet_20190524_vanilla_trans_128_2"
#python3 train_siamese_network_CV_v2.py --output_dim 128 --dir "$var1$dir_name" --epochs 25  --view 'trans' --unet > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

###########################################################
################## 512 full model #########################
###########################################################

var1="/storage/20190524_models_fullTrain/"

#dir_name="siamnet_20190522_vanilla_trans_full_128_REDO_f2"
#python3 train_siamese_network.py --output_dim 128 --dir "$var1$dir_name" --epochs 25  --view 'trans'  > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

dir_name="unet_20190522_vanilla_siamese_full_256_m"
python3 train_siamese_network.py --output_dim 256 --dir "$var1$dir_name" --epochs 20  --view 'siamese' --unet > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="siamnet_20190522_vanilla_sag_full_128_REDO"
#python3 train_siamese_network.py --output_dim 128 --dir "$var1$dir_name" --epochs 25  --view 'sag'  > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="unet_20190522_vanilla_trans_full_128_REDO"
#python3 train_siamese_network.py --output_dim 128 --dir "$var1$dir_name" --epochs 20  --view 'trans' --unet  > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="siamnet_20190522_vanilla_trans_full_128_REDO"
#python3 train_siamese_network.py --output_dim 128 --dir "$var1$dir_name" --epochs 20  --view 'trans'  > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"



#dir_name="unet_20190522_vanilla_sag_full_256_REDO2"
#python3 train_siamese_network.py --output_dim 256 --dir "$var1$dir_name" --epochs 20  --view 'sag' --unet  > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="unet_20190522_vanilla_sag_full_512_REDO2"
#python3 train_siamese_network.py --output_dim 512 --dir "$var1$dir_name" --epochs 20  --view 'sag' --unet  > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="unet_20190522_vanilla_sag_full_1024_REDO2"
#python3 train_siamese_network.py --output_dim 1024 --dir "$var1$dir_name" --epochs 20  --view 'sag' --unet  > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="unet_20190522_vanilla_sag_full_4096_REDO2"
#python3 train_siamese_network.py --output_dim 4096 --dir "$var1$dir_name" --epochs 20  --view 'sag' --unet  > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

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
>>>>>>> Stashed changes

#dir_name="unet_20190520_jigsaw_siamese"
#python3 train_siamese_network_CV_v2.py --dir "$var1$dir_name" --epochs 35  --view 'siamese' --unet --checkpoint ../JigsawPuzzlePytorch-master/checkpoints_20190415_bs256_e70_lr0.01_64_c1/jps_069_014770.pth.tar > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="siamnet_20190520_jigsaw_siamese"
#python3 train_siamese_network_CV_v2.py --dir "$var1$dir_name" --epochs 35  --view 'siamese' --checkpoint ../JigsawPuzzlePytorch-master/checkpoints_20190415_bs256_e70_lr0.01_64_c1/jps_069_014770.pth.tar > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"
