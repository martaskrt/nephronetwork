#### TODO
  * checkpoint restoration
  * fix tensorboard implementation on hpf
  * clean up code and documentation. right now it's more for my own purposes than anything!
  * fix the naming of the 'best' model checkpoint

__________________

## Kidney Label Classifier

Using images labelled from the Goldenberg labelling party consisting of roughly 23k labelled images - predict whether a given ultrasound corresponds to a particular view using CNNs.

* Images are split into a training/validation (80/20) cohorts by patient id as standard.
* Checkpoints are saved for the latest model (ie. if ran for 50 epochs, then a model will be saved and overwritten for each and every epoch, up until cancellation or epoch = 50) and a separate model corresponding to the lowest validation loss (the 'best' model).
* Dataset is divided into the 'full' dataset, and a smaller sample dataset consisting of 2000 images
* BCE loss can be weighted by the class count/min(class_count) or class count/max(class_count)


### Current Available Tasks:
* Bladder vs Other (Includes all other images ie. sagittal and transverse, *not just* images that have been labelled as 'other' ) (**2 - bladder**)
* Bladder vs Other vs Sagittal vs Transverse (**4 - view**)
* Bladder vs Other vs Sagittal Left vs Sagittal Right vs Transverse Left vs Transverse Right (**6 -- granular**)

This is passed as an argument to `train_and_eval.sh` as `task={bladder|view|granular}`

### Current Available Models:
#### Pre-Trained:
Feature Extraction -> Adaptive Pool -> Classifier
  * alexnet
  * vgg

No Changes:
  * densenet
  * squeezenet
  * resnet

#### Custom:
* typical conv-batch-relu-maxpool x 4 followed by average pool into a classifier
* filters used are k = 62, 128, 256 and 512, respectively
* max-pool used has stride = 2 and window = 3

This is passed as an argument to `train_and_eval.py` as `model={alexnet|vgg|densenet|squeezenet|resnet|custom}`.

Custom architectures can be easily added by modifying the torch.nn.module in `net.py` and the dictionary.

###  Folder Set-up
Folder directory should be set up as below.
```
nephro
└── nephro.sh
└── evaluate_models.sh
└── make_all_preds.sh
└── nephro_net
    └── scripts_in_this_repo
└── ./data
    ├── ./data/imgs [2000 entries exceeds filelimit, not opening dir]
    └── ./data/kidney_manifest.csv

```



### Scripts
`01_combine_make_new_labels.R`
* combines all the kidney labels into a single .csv
* creates labels for all views (granular), views (view), and bladder + other (bladder)
* outputs: `kidney_manifest.csv`, to be used in dataloader
* also takes a random 2000 images as a subset for testing purposes


`train_and_eval.py`
* the CNN workhorse

`evaluate_models.py`
* given models (checkpoints), loops over and generates various metrics and figures (AUC if binary, confusion matrix, and classification report) for training and validation sets
* technically not best practice for how confusion matrix is classified for binary tasks

`predict_bladder.py`
* computes the probability that a given image is a bladder

`get_preds.py`
* computes softmax (probabilities) that a given image belongs to a class


### Example Usage :

1a. Submit `nephro.sh` to scheduler on HPF
```
qsub -v task=bladder,model=alexnet,run=your_run_name nephro.sh
qsub -v task=view,model=alexnet,run=your_run_name nephro.sh
qsub -v task=granular,model=alexnet,run=hello_world nephro.sh
```

1b. Alternatively, run `train_and_eval.py` on your desktop/server/wherever:
```
python3 ~/PycharmProjects/nephro/train_and_eval.py \
      --root_path='/Users/delvin/Documents/Projects/nephro_test/' \
      --num_epochs=100 \
      --manifest_path='/Users/delvin/Documents/Projects/nephro_test/data/kidney_manifest.csv' \
      --model_out='/Users/delvin/Documents/Projects/nephro_test/output'   \
      --metrics_every_iter=10 \
      --no_wts \
      --batch_size=64 \
      --run_name=alexnet_no_wts
      --model=alexnet
      --task=granular
```

All scripts below can also be called simply by invoking `python3 script_name_here.py`

2.  Calculate performance metrics on the 'best' model (lowest validation loss)
* edit checkpoints list in `evaluate_models.py` to where your checkpoints were saved as well as the directories for the dataloader
* runs a loop over the checkpoints - for each checkpoint, using the training and validation set (or only one!)
  * if a binary task (bladder vs other), generates AUROC, plots the AUC, and saves it.
  * computes confusion matrix using the GREATEST probability of the two classes. ie. if p(y = 0) > p(y = 1) then the predicted class will be 'Other' and vice-versa. (not ideal but just to visualize how well the classifier does)
* for all other tasks and models, inclusive of bladder, confusion matrices and classification reports courtesy of sklearn.metrics will be generated and saved
```
qsub evaluate_models.sh

# calls evaluate_models.py
```

3) For all the other roughly 70k images (including both labelled and non-labelled images), make predictions using all the images
* outputs a csv where columns are the file name and the probability that the image is of a bladder.

```
qsub make_all_preds.sh

# calls predict_bladder.py
```

Important arguments:

```
(nnet) [delvinso@node415 nephro_net]$ python train_and_eval.py -h
usage: train_and_eval.py [-h] --root_path ROOT_PATH --manifest_path
                         MANIFEST_PATH [--task TASK] --model_out MODEL_OUT
                         [--model MODEL] --num_epochs NUM_EPOCHS
                         [--learning_rate LEARNING_RATE]
                         [--metrics_every_iter METRICS_EVERY_ITER] [--use_wts]
                         [--no_wts] [--run_suffix RUN_SUFFIX]
                         [--batch_size BATCH_SIZE] [--run_name RUN_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --root_path ROOT_PATH
                        root directory
  --manifest_path MANIFEST_PATH
                        absolute path of the manifest file
  --task TASK           one of granular (6), view (4), or bladder (2)
  --model_out MODEL_OUT
                        output directory
  --model MODEL         one of vgg, resnet, alexnet, squeezenet, densenet, or
                        custom
  --num_epochs NUM_EPOCHS
                        int, the number of epochs
  --learning_rate LEARNING_RATE
                        the learning rate
  --metrics_every_iter METRICS_EVERY_ITER
                        calculates metrics every i batches
  --use_wts
  --no_wts
  --run_suffix RUN_SUFFIX
                        suffix to append to end of tensorflow log file
  --batch_size BATCH_SIZE
                        batch size to use in training
  --run_name RUN_NAME   name of output directory where checkpoints and log is
                        saved
```

`--root_path`: should be set to `./nephro`.

`--model_out`: will automatically be created if it does not already exist. a single run will be output as `model_out/run_name/model_task` (double check this pls)

`--no_wts (default) | --use_wts`: will scale loss by the *training* datasets class balance

`--task`: one of granular (all 6 labels), view (4 labels, no L/R), or bladder (2 labels, bladder vs other)`

`--model`: one of vgg, resnet, alexnet, densenet, squeezenet or custom

