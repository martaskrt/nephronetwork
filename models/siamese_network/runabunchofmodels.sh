var1="mountpoint/20190520_models/"
var2=".txt"

dir_name="siamnet_20190520_vanilla_siamese"
python3 train_siamese_network_CV_v2.py --dir "$var1$dir_name" --epochs 35  --view 'siamese' > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

dir_name="siamnet_20190520_vanilla_sag"
python3 train_siamese_network_CV_v2.py --dir "$var1$dir_name" --epochs 35  --view 'sag' > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

dir_name="siamnet_20190520_vanilla_trans"
python3 train_siamese_network_CV_v2.py --dir "$var1$dir_name" --epochs 35  --view 'trans' > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="unet_20190520_vanilla_sag"
#python3 train_siamese_network_CV_v2.py --dir "$var1$dir_name" --epochs 35  --view 'sag' --unet > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

#dir_name="unet_20190520_vanilla_trans"
#python3 train_siamese_network_CV_v2.py --dir "$var1$dir_name" --epochs 35  --view 'trans' --unet > "$dir_name$var2" && mv "$dir_name$var2" "$var1$dir_name"

