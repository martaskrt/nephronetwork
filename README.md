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


Moving text files to result directory 
Delete stage one and stage 2 
Delete: 
SiameseNetworkSkipConnects.py 
TriameseNetwork.py
cuda-repo-ubuntu1604_9.2.88-1_amd64.deb
Train_triamese_network.py
train_triamese_network_CV.py
Cross validation versus no cross validation 
Transfer to transfer learning 
Transfer visualisation to post process 

Crunch results 
Fins the best epoc to stop on 
Computes the AUC and AUPRC from the results/


analyzing-output-pths.py
After best epoc, accuracy and details of each person (prob of surgery) can be found. 
For each folds, create a csv file and then the file is read to R to make the AUC and scatter plots 

