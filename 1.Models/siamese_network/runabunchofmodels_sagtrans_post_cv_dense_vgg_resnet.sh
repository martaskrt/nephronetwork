#!/bin/bash


echo "vgg_bn sag running"
python3 train_resnet-vgg_CV_v2.py --vgg_bn --git_dir /home/lauren/ --view sag --stop_epoch 24 --lr 0.001 --batch_size 64 --dir /storage/dense-res-vgg-output-pths/vgg_bn/sag/ > vgg_bn_sag_lr0.001_bs64-final.txt
echo "vgg_bn sag complete"

echo "densenet sag running"
python3 train_resnet-vgg_CV_v2.py --densenet --git_dir /home/lauren/ --view sag  --stop_epoch 20 --lr 0.001 --batch_size 64 --dir /storage/dense-res-vgg-output-pths/cv/densenet/sag/ > densenet_sag_lr0.001_bs64-final.txt
echo "densenet sag complete"

echo "resnet18 sag running"
python3 train_resnet-vgg_CV_v2.py --resnet18 --git_dir /home/lauren/ --view sag --stop_epoch 26 --lr 0.001 --batch_size 64 --dir /storage/dense-res-vgg-output-pths/cv/resnet/sag/ > resnet18_sag_lr0.001_bs64-final.txt
echo "resnet18 sag complete"

echo "ALL SAG MODELS COMPLETE"

echo "vgg_bn trans running"
python3 train_resnet-vgg_CV_v2.py --vgg_bn --git_dir /home/lauren/ --view trans  --stop_epoch 22 --lr 0.001 --batch_size 64 --dir /storage/dense-res-vgg-output-pths/cv/vgg_bn/trans/ > vgg_bn_trans_lr0.001_bs64-final.txt
echo "vgg_bn trans complete"

echo "densenet trans running"
python3 train_resnet-vgg_CV_v2.py --densenet --git_dir /home/lauren/ --view trans --stop_epoch 20 --lr 0.001 --batch_size 64 --dir /storage/dense-res-vgg-output-pths/cv/densenet/trans/ > densenet_trans_lr0.001_bs64-final.txt
echo "densenet trans complete"

echo "resnet18 trans running"
python3 train_resnet-vgg_CV_v2.py --resnet18 --git_dir /home/lauren/ --view trans --stop_epoch 17 --lr 0.001 --batch_size 64 --dir /storage/dense-res-vgg-output-pths/cv/resnet/trans/ > resnet18_trans_lr0.001_bs64-final.txt
echo "resnet18 trans complete"

echo "ALL MODELS COMPLETE"

