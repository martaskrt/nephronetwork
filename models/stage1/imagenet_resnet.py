import keras
from keras.applications import resnet50
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import numpy as np
import os
import pydicom
from skimage import transform
from skimage.color import rgb2gray

#Load the ResNet50 model
resnet_model = resnet50.ResNet50(weights='imagenet')

PathDicom = '../../data/temp/'
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

for image in lstFilesDCM:
    ds = pydicom.dcmread(image)
    img = ds.pixel_array.astype('float') # extract the image info from DICOM file to np array

    if len(img.shape) == 3:
        img = rgb2gray(img)

    img = transform.resize(img, (224,224)) # resize to ResNet trained network input size
    # plt.imshow(img, cmap='gray')
    # plt.show()

    img = np.expand_dims(img, axis=0) # adds a batch dimension
    img = np.repeat(img[..., np.newaxis], 3, -1) # copy the grayscale to 3 RGB columns
    proc = resnet50.preprocess_input(img)

    pred = resnet_model.predict(img)

    label = decode_predictions(pred) # inputs are (batch_size, x_coord, y_coord, colors) ie, (1,224,224,3)
    print(label)

