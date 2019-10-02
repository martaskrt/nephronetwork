#!/bin/bash


echo "vgg_bn sag running"
python3 train_resnet-vgg_CV_v2.py --vgg_bn --git_dir /home/lauren/ --view sag --lr 0.001 --pretrained --batch_size 64 --cv --dir /storage/dense-res-vgg-output-pths/cv/pretrained/vgg_bn/sag/ > vgg_bn_cv_pt_sag_lr0.001_bs64.txt
echo "vgg_bn sag complete"

echo "densenet sag running"
python3 train_resnet-vgg_CV_v2.py --densenet --git_dir /home/lauren/ --view sag --lr 0.001 --pretrained  --batch_size 64 --cv --dir /storage/dense-res-vgg-output-pths/cv/pretrained/densenet/sag/ > densenet_cv_pt_sag_lr0.001_bs64.txt
echo "densenet sag complete"

echo "resnet18 sag running"
python3 train_resnet-vgg_CV_v2.py --resnet18 --git_dir /home/lauren/ --view sag --lr 0.001 --pretrained  --batch_size 64 --cv --dir /storage/dense-res-vgg-output-pths/cv/pretrained/resnet/sag/ > resnet18_cv_pt_sag_lr0.001_bs64.txt
echo "resnet18 sag complete"

echo "ALL SAG MODELS COMPLETE"

echo "vgg_bn trans running"
python3 train_resnet-vgg_CV_v2.py --vgg_bn --git_dir /home/lauren/ --view trans --lr 0.001 --pretrained  --batch_size 64 --cv --dir /storage/dense-res-vgg-output-pths/cv/pretrained/vgg_bn/trans/ > vgg_bn_cv_pt_trans_lr0.001_bs64.txt
echo "vgg_bn trans complete"

echo "densenet trans running"
python3 train_resnet-vgg_CV_v2.py --densenet --git_dir /home/lauren/ --view trans --lr 0.001 --pretrained  --batch_size 64 --cv --dir /storage/dense-res-vgg-output-pths/cv/pretrained/densenet/trans/ > densenet_cv_pt_trans_lr0.001_bs64.txt
echo "densenet trans complete"

echo "resnet18 trans running"
python3 train_resnet-vgg_CV_v2.py --resnet18 --git_dir /home/lauren/ --view trans --lr 0.001 --pretrained  --batch_size 64 --cv --dir /storage/dense-res-vgg-output-pths/cv/pretrained/resnet/trans/ > resnet18_cv_pt_trans_lr0.001_bs64.txt
echo "resnet18 trans complete"

echo "ALL TRANS MODELS COMPLETE"

echo "vgg_bn siamese running"
python3 train_resnet-vgg_CV_v2.py --vgg_bn --git_dir /home/lauren/ --view siamese --lr 0.001 --pretrained  --batch_size 64 --cv --dir /storage/dense-res-vgg-output-pths/cv/pretrained/vgg_bn/siamese/ > vgg_bn_cv_pt_siamese_lr0.001_bs64.txt
echo "vgg_bn siamese complete"

echo "densenet siamese running"
python3 train_resnet-vgg_CV_v2.py --densenet --git_dir /home/lauren/ --view siamese --lr 0.001 --pretrained  --batch_size 64 --cv --dir /storage/dense-res-vgg-output-pths/cv/pretrained/densenet/siamese/ > densenet_cv_pt_siamese_lr0.001_bs64.txt
echo "densenet siamese complete"

echo "resnet18 siamese running"
python3 train_resnet-vgg_CV_v2.py --resnet18 --git_dir /home/lauren/ --view siamese --lr 0.001 --pretrained  --batch_size 64 --cv --dir /storage/dense-res-vgg-output-pths/cv/pretrained/resnet/siamese/ > resnet18_cv_pt_siamese_lr0.001_bs64.txt
echo "resnet18 siamese complete"

echo "ALL MODELS COMPLETE"

