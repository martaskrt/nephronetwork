var1="mountpoint/20190520_models/"
var2=".txt"


#dir_name="unet_20190520_vanilla_sag"
#python3 train_siamese_network_CV_v2.py --dir "$var1$dir_name" --epochs 35  --view 'sag' --unet > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="unet_20190520_vanilla_trans"
#python3 train_siamese_network_CV_v2.py --dir "$var1$dir_name" --epochs 35  --view 'trans' --unet > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="unet_20190520_jigsaw_siamese"
#python3 train_siamese_network_CV_v2.py --dir "$var1$dir_name" --epochs 35  --view 'siamese' --unet --checkpoint ../JigsawPuzzlePytorch-master/checkpoints_20190415_bs256_e70_lr0.01_64_c1/jps_069_014770.pth.tar > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

dir_name="siamnet_20190520_jigsaw_siamese"
python3 train_siamese_network_CV_v2.py --dir "$var1$dir_name" --epochs 35  --view 'siamese' --checkpoint ../JigsawPuzzlePytorch-master/checkpoints_20190415_bs256_e70_lr0.01_64_c1/jps_069_014770.pth.tar > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"
