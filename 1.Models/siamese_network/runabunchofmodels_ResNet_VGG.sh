#!/bin/bash

declare -a my_lrs=("0.001"  "0.005")
declare -a my_views=("trans" "sag" "siamese")
declare -a my_bs=(16 128)

###########################################################
######                MODEL LOOP                    #######
###########################################################

for lr in "${my_lrs[@]}"
do
    for view in "${my_views[@]}"
    do
        for bs in "${my_bs[@]}"
        do
            python3 train_resnet-vgg_CV_v2.py --vgg_bn --git_dir /home/lauren/ --view $view --lr $lr --batch_size $bs --cv > /storage/vgg-densenet-out/vgg_bn-lr${lr}-bs${bs}-${view}.txt
	        python3 train_resnet-vgg_CV_v2.py --densenet --git_dir /home/lauren/ --view $view --lr $lr --batch_size $bs --cv > /storage/vgg-densenet-out/vgg_bn-lr${lr}-bs${bs}-${view}.txt
        done
    done
done
