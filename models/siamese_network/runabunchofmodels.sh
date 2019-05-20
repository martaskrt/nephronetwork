var1="mountpoint/"
var2=".txt"
dir_name="unet_20190517_vanilla_CV_crop0_OR_plusRef_BN_cwpen2"
python3 train_siamese_network_CV_v2.py --dir "$dir_name" --epochs 35  --view 'siamese' --crop 0 --unet > "$dir_name$var2" && mv "$dir_name$var2" "$dir_name"


