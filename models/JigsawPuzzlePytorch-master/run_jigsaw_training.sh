IMAGENET_FOLD="/home/marta/nephronetwork-github/nephronetwork/models/JigsawPuzzlePytorch-master/ILSVRC2012_img_train"

GPU=${1} # gpu used
CHECKPOINTS_FOLD=${2} #path_to_output_folder
rootname="checkpoints_20190415_bs256_e70_lr0.01_50_c1_16p"
ext=".txt"
#python JigsawTrain.py ${IMAGENET_FOLD} --checkpoint=${CHECKPOINTS_FOLD} \
#                      --classes=1000 --batch 128 --lr=0.001 --gpu=${GPU} --cores=10
python3 JigsawTrain.py ${IMAGENET_FOLD} --classes=1000 --cores=10 --checkpoint $rootname --lr=0.01 --epochs 70 > "$rootname$ext" && mv "$rootname$ext" $rootname
