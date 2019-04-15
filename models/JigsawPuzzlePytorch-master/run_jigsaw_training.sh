IMAGENET_FOLD="/home/marta/nephronetwork/models/JigsawPuzzlePytorch-master/ILSVRC2012_img_train"

GPU=${1} # gpu used
CHECKPOINTS_FOLD=${2} #path_to_output_folder

#python JigsawTrain.py ${IMAGENET_FOLD} --checkpoint=${CHECKPOINTS_FOLD} \
#                      --classes=1000 --batch 128 --lr=0.001 --gpu=${GPU} --cores=10
python3 JigsawTrain.py ${IMAGENET_FOLD} --classes=1000 --batch 128 --lr=0.001 --cores=10 --epochs 2000 > 20190330_JIGSAWTRAINING_kideyimagse_2000e_bs128_cl1000_lr0.001.txt
