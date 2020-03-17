# nephronetwork
### 1. Pre-process
Out put is a pickle file
Crop
Normalizing resolution
main files:
* Extract_labels.py

Cleaning labels in .csv files (actual data without images). It maps patient identifier to a number

* Load_dataset.py  
Normalises the contrast and organizes the views. Splits data to train and test. Pulls out the cropping parameter. Cn crop in the bottom. Pull out etiology.
It is called only in the training

*Preprocess.py
calls two other scripts above

* Preprocess_full_us
Unzip cab files pull in each DCOM files. Crops them. Sets the contrast. Converts to JPEG and save the .pickle file.
Similar to preprocess.py.


### 2. Pre-training
Ask Carson
- MNIST
- OCT
- Puzzle  
- Sequence of altrasounds


### 3. Model development

#### 3.1 Traditional computer vision
* SVM


#### 3.2 Neural Networks
* Siamese
* UNet


### 4. Post process

main files:
* Crunch results
Fins the best epoc to stop on
Computes the AUC and AUPRC from the results/


* analyzing-output-pths.py
After best epoc, accuracy and details of each person (prob of surgery) can be found.
For each folds, create a csv file and then the file is read to R to make the AUC and scatter plots



___________________________________

## Kidney Label Classifier

*NOTE: additional documentation can be found in kidney_label_classifier/*

Using images labelled from the Goldenberg labelling party consisting of roughly 23k labelled images - predict whether a given ultrasound corresponds to a particular view using CNNs.

* Images are split into a training/validation (80/20) cohorts by patient id as standard.
* Checkpoints are saved for the latest model (ie. if ran for 50 epochs, then a model will be saved and overwritten for each and every epoch, up until cancellation or epoch = 50) and a separate model corresponding to the lowest validation loss (the 'best' model).
* Dataset is divided into the 'full' dataset, and a smaller sample dataset consisting of 2000 images
* BCE loss can be weighted by the class count/min(class_count) or class count/max(class_count)

Example usage:
```
python3 /home/delvinso/nephro/nephro_net/train_and_eval.py \
  --root_path='/home/delvinso/nephro/' \
  --num_epochs=100  \
  --manifest_path='/home/delvinso/nephro/data/kidney_manifest.csv'  \
  --model_out='/home/delvinso/nephro/output' \
  --metrics_every_iter=100 \
  --no_wts \
  --batch_size=16 \
  --run_name=alexnet_no_wts
  --model=alexnet
  --task=granular
```
