#!/bin/bash


python3 train_resnet-vgg_CV_v2.py --vgg_bn --git_dir /home/lauren/ --view sag --lr 0.001 --batch_size 64 --cv --dir /storage/dense-res-vgg-output-pths/cv/vgg_bn/sag/ > vgg_bn_cv_sag_lr0.001_bs64.txt
python3 train_resnet-vgg_CV_v2.py --densenet --git_dir /home/lauren/ --view sag --lr 0.001 --batch_size 64 --cv --dir /storage/dense-res-vgg-output-pths/cv/densenet/sag/ > densenet_cv_sag_lr0.001_bs64.txt
python3 train_resnet-vgg_CV_v2.py --resnet18 --git_dir /home/lauren/ --view sag --lr 0.001 --batch_size 64 --cv --dir /storage/dense-res-vgg-output-pths/cv/resnet/sag/ > resnet18_cv_sag_lr0.001_bs64.txt

python3 train_resnet-vgg_CV_v2.py --vgg_bn --git_dir /home/lauren/ --view trans --lr 0.001 --batch_size 64 --cv --dir /storage/dense-res-vgg-output-pths/cv/vgg_bn/trans/ > vgg_bn_cv_sag_lr0.001_bs64.txt
python3 train_resnet-vgg_CV_v2.py --densenet --git_dir /home/lauren/ --view trans --lr 0.001 --batch_size 64 --cv --dir /storage/dense-res-vgg-output-pths/cv/densenet/trans/ > densenet_cv_sag_lr0.001_bs64.txt
python3 train_resnet-vgg_CV_v2.py --resnet18 --git_dir /home/lauren/ --view trans --lr 0.001 --batch_size 64 --cv --dir /storage/dense-res-vgg-output-pths/cv/resnet/trans/ > resnet18_cv_sag_lr0.001_bs64.txt
