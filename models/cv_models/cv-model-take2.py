# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 18:44:27 2019

@author: Lauren
"""
####
###         IMPORT LIBRARIES
####

import numpy as np
import mahotas
import cv2
import sys
sys.path.insert(0, 'C:/Users/larun/Desktop/Data Science Core/Projects/Urology/nephronetwork/preprocess/')
import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

## following: https://gogul09.github.io/software/image-classification-python

####
###         CV FUNCTIONS
####

# rescale image intensities to 255
def rescale_img(image):
    image=image*255
    return image

def img_int(image):
    image=np.rint(image)
    image = image.astype(int)
    return(image)


# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(image).mean(axis=0)
    return haralick

# get vector of labels (back) and matching feature vector for classifier
def get_features_and_labels(y_array,x_array,getCols = True):
    # empty lists to hold feature vectors and labels
    global_features = []
    labels = []
    
    num = 0
    # loop over the training data sub-folders
    while num < (len(x_array) - 1):
        fv_hu_moments = []
        fv_haralick = []
        
        for count in range(2):    
            # get the current image
            current_img = x_array[num]
            # rescale image to 255
            current_img = rescale_img(current_img)            
            
            ####################################
            # Global Feature extraction
            ####################################
            fv_hu_moments.append(fd_hu_moments(current_img))
            fv_haralick.append(fd_haralick(img_int(current_img)))
            
            ###################################
            # Concatenate global features
            ###################################
            global_feature = np.hstack([fv_haralick, fv_hu_moments]).flatten().tolist()
            
            num = num + 1
            
        # update the list of labels and feature vectors
        #labels.append(current_label)
        
        global_features.append(global_feature)
        labels.append(y_array[num-1])
        
    global_features = StandardScaler().fit_transform(global_features)
    
    if getCols:
        col_vec = []
        for i in range(len(labels)):
            if labels[i] == 1:
                col_vec.append('r')
            else:
                col_vec.append('b')
        return labels,global_features,col_vec
    
    else:
        return labels,global_features

#pc_mat=principalComponents.reshape(10,-1)
def plot_pcs(pc_mat,n_pcs,train_labels,col_vec,plot_name=""):    
    # Create plot
    fig = plt.figure()
    
    for k in range(n_pcs):
        ax = fig.add_subplot(n_pcs, 1, k+1)
         
        for i in range(len(train_labels)):
    #    for i in np.nditer(np.where(np.array(train_labels) == 0)):
    #    for i in np.nditer(np.where(np.array(train_labels) == 1)):
            x, y = pc_mat[[k,k+1],i]
            ax.scatter(x, y, alpha=0.8, c=col_vec[i], edgecolors='none', s=30)
         
        plt.title(plot_name)
        #plt.legend(loc=2)
    plt.show()


####
###         CV ANALYSIS    
####

## Import data
train_X, train_y, train_clin_features, test_X, test_y, test_clin_features = load_dataset.load_dataset(pickle_file="C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Image-analysis/preprocessed_images_20190315.pickle",get_cov=True)

## Extract features    
    #    Train
train_labels,train_img_features,train_col_vec = get_features_and_labels(train_y,train_X)
    #    Test
test_labels,test_img_features,test_col_vec = get_features_and_labels(test_y,test_X)

    ##  PCA OF CV FEATURES

## Extracting PCs
n_pcs = 9
pca = PCA(n_components=n_pcs)
principalComponents = pca.fit_transform(train_img_features)

## Plot PCs -- color by machine 
plot_pcs(principalComponents.reshape(n_pcs,-1),n_pcs = n_pcs,train_labels = train_labels,col_vec = train_col_vec)

    ## CLASSIFIERS USING CV FEATURES
    
# Random Forest - add age, sex, etc features?
clf=RandomForestClassifier(n_estimators=1000,max_depth=12)    
clf.fit(train_img_features,train_labels)
train_pred = clf.predict(train_img_features)
print("Accuracy:",metrics.accuracy_score(train_labels, train_pred))
metrics.confusion_matrix(train_labels, train_pred)
#plt.plot(train_labels,train_pred,'bo')

test_pred = clf.predict(test_img_features)
print("Accuracy:",metrics.accuracy_score(test_labels, test_pred))
metrics.confusion_matrix(test_labels, test_pred)
#plt.plot(test_labels,test_pred,'bo')


# SVM
    ## RBF kernel SVM: low accuracy
clf = svm.SVC(gamma='scale')
clf.fit(train_img_features, train_labels)
train_pred = clf.predict(train_img_features)
print("Accuracy:",metrics.accuracy_score(train_labels, train_pred))
metrics.confusion_matrix(train_labels, train_pred)

    ## Poly kernel SVM: low accuracy, only slightly better than RBF
clf = svm.SVC(gamma='scale',kernel = 'poly')
clf.fit(train_img_features, train_labels)
train_pred = clf.predict(train_img_features)
print("Accuracy:",metrics.accuracy_score(train_labels, train_pred))
metrics.confusion_matrix(train_labels, train_pred)

# Gradient boosting classifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                 max_depth=1, random_state=0).fit(train_img_features, train_labels)
train_pred = clf.predict(train_img_features)
print("Accuracy:",metrics.accuracy_score(train_labels, train_pred))
metrics.confusion_matrix(train_labels, train_pred)


