#!/bin/bash
#PBS -q gpu -l nodes=1:ppn=1:gpus=1,vmem=60g,mem=60g,walltime=12:00:00

module load anaconda
module load glibc
source activate nnet
model=${model}
task=${task}
run_name=${run}
echo "Task: ${task}"
echo "Model: ${model}"
python3 /home/delvinso/nephro/nephro_net/train_and_eval.py \
  --root_path='/home/delvinso/nephro/' \
  --num_epochs=100  \
  --manifest_path='/home/delvinso/nephro/data/kidney_manifest.csv'  \
  --model_out='/home/delvinso/nephro/output' \
  --metrics_every_iter=100 \
  --no_wts \
  --batch_size=16 \
  --run_name=${run_name} \
  --task=${task} \
  --model=${model}

# example usage
# qsub -v task=bladder,model=alexnet,run=your_run_name nephro.sh
# qsub -v task=view,model=alexnet,run=your_run_name nephro.sh
# qsub -v task=granular,model=alexnet,run=hello_world nephro.sh
