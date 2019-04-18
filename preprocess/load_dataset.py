import pandas as pd
import math
from skimage import exposure
import numpy as np
import ast
from numpy import genfromtxt
import matplotlib.pyplot as plt


def open_file(file):
    print(file)
    data = pd.read_pickle(file)
    #data = pd.read_pickle("preprocessed_images.pickle")
    return data


def train_test_split(patient_ids_sorted, split):
    train_split = math.floor(len(patient_ids_sorted)*split)
    train_ids = list(map(lambda x: x[0], patient_ids_sorted[:train_split]))
    test_ids = list(map(lambda x: x[0], patient_ids_sorted[train_split:]))
    return train_ids, test_ids


def get_y(data, siamese=False, samples_to_exclude=None):
    labels = []
    if siamese:
        num_samples = len(data)
        for i in range(num_samples):
            if i in samples_to_exclude:
                continue
            if data[i]['surgery'].iloc[0] == 0:
                labels.append(0)
            else:
                if data[i]['kidney_side'].iloc[0] == "Left" and data[i]['hydro_kidney'].iloc[0] == "Left":
                    labels.append(1)
                elif data[i]['kidney_side'].iloc[0] == "Right" and data[i]['hydro_kidney'].iloc[0] == "Right":
                    labels.append(1)
                else:
                    labels.append(0)
    else:
        num_samples = data.shape[0]

        for i in range(num_samples):
            if data.iloc[i]['surgery'] == 0:
                labels.append(0)
            else:
                if data.iloc[i]['kidney_side'] == "Left" and data.iloc[i]['hydro_kidney'] == "Left":
                    labels.append(1)
                elif data.iloc[i]['kidney_side'] == "Right" and data.iloc[i]['hydro_kidney'] == "Right":
                    labels.append(1)
                else:
                    labels.append(0)
    return labels


def get_X(data, contrast, image_dim, siamese=False):

    if siamese:
        num_samples = len(data)
        X = np.zeros((num_samples, 2, image_dim, image_dim))
    else:
        num_samples = data.shape[0]
        X = np.zeros((num_samples, image_dim, image_dim))
    X = []
    samples_to_exclude = []
    if siamese:
        for i in range(num_samples):
            if (data[i]['image'].shape[0] < 2):
                samples_to_exclude.append(i)
                continue
            group = np.zeros((2, image_dim, image_dim))
            for j in range(2):

                if data[i]['kidney_view'].iloc[j] == "Sag":
                    idx = 0
                elif data[i]['kidney_view'].iloc[j] == "Trans":
                    idx = 1
                image = data[i]['image'].iloc[j].reshape((image_dim,image_dim))
                if contrast == 0:
                    group[idx, :, :] = image
                elif contrast == 1:
                    group[idx, :, :] = exposure.equalize_hist(image)
                elif contrast == 2:
                    group[idx, :, :] = exposure.equalize_adapthist(image)
                elif contrast == 3:
                    group[idx, :, :] = exposure.rescale_intensity(image)
            X.append(group)
        X = np.array(X)
        return X, samples_to_exclude
    else:
        for i in range(num_samples):
            image = data.iloc[i]['image'].reshape((image_dim,image_dim))
            if contrast == 0:
                X.append(image)
            elif contrast == 1:
                X.append(exposure.equalize_hist(image))
            elif contrast == 2:
                X.append(exposure.equalize_adapthist(image))
            elif contrast == 3:
                X.append(exposure.rescale_intensity(image))
        X = np.array(X)
        return X

'''

'''
def get_f(data):
    features = {}

    for column in data.columns:

        if column not in ['laterality', 'surgery', 'crop_style', 'hydro_kidney', 'image', 'kidney_view',
                          'kidney_side']:
            features[column] = []

    features["saggital"] = []
    features["male"] = []

    if 'sample_num' in features:
        features['total_patient_samples'] = []
        total_patient_samples = list(set(zip(data.study_id, data.sample_num)))
        id2numsamples = {}  # map study_id to total_num of samples
        for i in total_patient_samples:
            id2numsamples[i[0]] = i[1]

    for i in range(data.shape[0]):
        for j in features:
            if j == "saggital":
                if data.iloc[i]['kidney_view'] == "Sag":
                    features[j].append(0)
                elif data.iloc[i]['kidney_view'] == "Trans":
                    features[j].append(1)
            elif j == "male":
                if data.iloc[i]['gender'] == "Male":
                    features[j].append(0)
                elif data.iloc[i]['gender'] == "Female":
                    features[j].append(1)
            elif j != 'total_patient_samples':
                features[j].append(data.iloc[i][j])
            else:
                study_id = data.iloc[i]['study_id']
                features[j].append(id2numsamples[study_id])
    return features


def load_train_test_sets(data, sort_by_date, split, contrast, image_dim, get_features, get_cov=False, siamese=False):
    patient_ids_and_ultrasound_dates = list(set(zip(data.study_id, data.date_of_ultrasound_1)))
    if sort_by_date:
        patient_ids_sorted = sorted(patient_ids_and_ultrasound_dates, key=lambda x: x[1])
    else:
        patient_ids_sorted = sorted(patient_ids_and_ultrasound_dates)
    train_ids, test_ids = train_test_split(patient_ids_sorted, split)
    train_data = data[data.study_id.isin(train_ids)]
    test_data = data[data.study_id.isin(test_ids)]

    if siamese:
        train_grouped = train_data.groupby(['study_id', 'sample_num', 'kidney_side'])
        train_groups = []
        for name, group in train_grouped:
            train_groups.append(group)

        train_X, samples_to_exclude = get_X(train_groups, contrast, image_dim, siamese=True)
        train_y = get_y(train_groups, siamese=True, samples_to_exclude=samples_to_exclude)

        test_grouped = test_data.groupby(['study_id', 'sample_num', 'kidney_side'])
        test_groups = []
        for name, group in test_grouped:
            test_groups.append(group)
        test_X, samples_to_exclude = get_X(test_groups, contrast, image_dim, siamese=True)
        test_y = get_y(test_groups, siamese=True, samples_to_exclude=samples_to_exclude)


    else:
        train_y = get_y(train_data)
        train_X = get_X(train_data, contrast, image_dim)

        test_y = get_y(test_data)
        test_X = get_X(test_data, contrast, image_dim)

    if get_features or get_cov:
        train_features = get_f(train_data)
        test_features = get_f(test_data)
        if get_cov:
            print(train_features.keys())
            train_cov = [train_features["study_id"], train_features["age_at_baseline"], train_features["male"], train_features["saggital"]]
            test_cov = [train_features["study_id"], test_features["age_at_baseline"], test_features["male"], test_features["saggital"]]
            train_features = train_cov
            test_features = test_cov
        return train_X, train_y, train_features, test_X, test_y, test_features

    else:
        return train_X, train_y, test_X, test_y


def get_sag(data, sort_by_date, split, contrast, image_dim, get_features, get_cov=False):
    data = data[data.kidney_view == "Sag"]
    return load_train_test_sets(data, sort_by_date, split, contrast, image_dim, get_features, get_cov=get_cov,)


def get_trans(data, sort_by_date, split, contrast, image_dim, get_features, get_cov=False):
    data = data[data.kidney_view == "Trans"]
    return load_train_test_sets(data, sort_by_date, split, contrast, image_dim, get_features, get_cov=get_cov,)

def get_siamese(data, sort_by_date, split, contrast, image_dim, get_features, get_cov=False):
    return load_train_test_sets(data, sort_by_date, split, contrast, image_dim, get_features, get_cov=get_cov, siamese=True)

def load_dataset(split=0.8, sort_by_date=True, contrast=0, drop_bilateral=True,
                         crop=0, get_features=False, image_dim=256, views_to_get="all", get_cov=False, pickle_file=""):

    data = open_file(pickle_file)

    if drop_bilateral:
        data = data[((data['hydro_kidney'] == data['kidney_side']) & (data['laterality'] == "Bilateral"))
                    | (data['laterality'] != "Bilateral")]
    data = data[data.crop_style == float(crop)]

    if views_to_get == "sag":
        return get_sag(data, sort_by_date, split, contrast, image_dim, get_features)
    elif views_to_get == "trans":
        return get_trans(data, sort_by_date, split, contrast, image_dim, get_features)
    elif views_to_get == "siamese":
        return get_siamese(data, sort_by_date, split, contrast, image_dim, get_features, get_cov)



def view_images(imgs, num_images_to_view=5, views_to_get="siamese"):
    counter = 0
    if views_to_get=="siamese":
        for img in imgs:
            if counter >= num_images_to_view:
                break
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(img[0], cmap='gray')
            plt.subplot(1, 2, 2)
            plt.imshow(img[1], cmap='gray')
            plt.show()
            counter += 1
    else:
        for img in imgs:
            if counter >= num_images_to_view:
                break
            plt.figure()
            plt.subplot(1, 1, 1)
            plt.imshow(img[0], cmap='gray')
            counter += 1

# datafile = "preprocessed_images_20190315.pickle"
# train_X, train_y, test_X, test_y = load_dataset(views_to_get="siamese", pickle_file=datafile)
#
# view_images(test_X)