import pandas as pd
import math
from skimage import exposure
import numpy as np
#import ast
#from numpy import genfromtxt
#import matplotlib.pyplot as plt


def open_file(file):
    #print(file)
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
            if (data[i]['kidney_view'].iloc[0] == "Sag" and data[i]['kidney_view'].iloc[1] == "Trans") or \
                    (data[i]['kidney_view'].iloc[1] == "Sag" and data[i]['kidney_view'].iloc[0] == "Trans"):
                pass
            else:
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
def get_f(data, samples_to_exclude=None, siamese=False):
    # study_id_date_map = pd.read_csv("../../preprocess/samples_with_studyids_and_usdates.csv")
    "/Volumes/terminator/nephronetwork/preprocess/"
    study_id_date_map = pd.read_csv("/Volumes/terminator/nephronetwork/preprocess/samples_with_studyids_and_usdates.csv")
    features = {}
    if siamese:
        for column in data[0]:

            if column not in ['laterality', 'surgery', 'crop_style', 'hydro_kidney', 'image', 'kidney_view']:
                features[column] = []
    else:
        for column in data.columns:
            if column not in ['laterality', 'surgery', 'crop_style', 'hydro_kidney', 'image', 'kidney_view']:
                features[column] = []
    features["saggital"] = []
    features["male"] = []
    features["sample_us_date"] = []


    # if 'sample_num' in features:
    #     features['total_patient_samples'] = []
    #     total_patient_samples = list(set(zip(data.study_id, data.sample_num)))
    #     id2numsamples = {}  # map study_id to total_num of samples
    #     for i in total_patient_samples:
    #         id2numsamples[i[0]] = i[1]
    if siamese:
        num_samples = len(data)
        for i in range(num_samples):
            if i in samples_to_exclude:
                continue
            for j in features:
                if j == "saggital":
                    if data[i]['kidney_view'].iloc[0] == "Sag":
                        features[j].append(0)
                    elif data[i]['kidney_view'].iloc[0] == "Trans":
                        features[j].append(1)
                elif j == "male":
                    if data[i]['gender'].iloc[0] == "Male":
                        features[j].append(0)
                    elif data[i]['gender'].iloc[0] == "Female":
                        features[j].append(1)
                elif j == "manufacturer":
                    manu = data[i][j].iloc[0].lower().replace("_", " ")
                    manu = manu.replace(".", " ")
                    manu = manu.replace(",", " ")
                    features[j].append('-'.join(manu.split()))

                elif j == "sample_num":
                    sample_num = int(data[i][j].iloc[0])
                    features[j].append(sample_num)
                    us_num = "date_us" + str(sample_num)
                    study_id = data[i]['study_id'].iloc[0]
                    us_date = str(study_id_date_map.loc[study_id_date_map['study_id'] == int(study_id)][us_num]).split("\n")[0].split()[1]

                    features['sample_us_date'].append(us_date)

                elif j != 'total_patient_samples' and j != "sample_us_date":
                    features[j].append(data[i][j].iloc[0])
                # else:
                #     study_id = data[i]['study_id'].iloc[0]
                #     features[j].append(id2numsamples[study_id])_

    else:
        num_samples = data.shape[0]
        for i in range(num_samples):
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
                elif j == "manufacturer":
                    manu = data.iloc[i][j].lower().replace("_", " ")
                    manu = manu.replace(".", " ")
                    manu = manu.replace(",", " ")
                    features[j].append('-'.join(manu.split()))
                elif j == "sample_num":
                    sample_num = int(data.iloc[i][j])
                    features[j].append(sample_num)
                    us_num = "date_us" + str(sample_num)
                    features['sample_us_date'].append(
                        study_id_date_map.loc[study_id_date_map['study_id'] == data.iloc[i]['study_id']][us_num])

                elif j != 'total_patient_samples' and j != "sample_us_date":
                    features[j].append(data.iloc[i][j])
                # else:
                #     study_id = data.iloc[i]['study_id']
                #     features[j].append(id2numsamples[study_id])
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
        assert len(train_y) == len(train_X)
        if get_features or get_cov:
            train_features = get_f(train_groups, samples_to_exclude=samples_to_exclude, siamese=True)
            if get_cov:
                train_cov = [train_features["study_id"], train_features["age_at_baseline"], train_features["male"],
                             train_features["saggital"], train_features["sample_num"], train_features['kidney_side'],
                             train_features["date_of_ultrasound_1"], train_features['sample_us_date'], train_features['manufacturer']]
                for item in train_cov:
                    assert len(item) == len(train_y)
                train_features = train_cov
        test_grouped = test_data.groupby(['study_id', 'sample_num', 'kidney_side'])
        test_groups = []
        for name, group in test_grouped:
            test_groups.append(group)
        test_X, samples_to_exclude = get_X(test_groups, contrast, image_dim, siamese=True)
        test_y = get_y(test_groups, siamese=True, samples_to_exclude=samples_to_exclude)
        assert len(test_y) == len(test_X)

        if get_features or get_cov:
            test_features = get_f(test_groups, samples_to_exclude=samples_to_exclude, siamese=True)
            if get_cov:
                test_cov = [test_features["study_id"], test_features["age_at_baseline"], test_features["male"],
                            test_features["saggital"], test_features["sample_num"], test_features['kidney_side'],
                            test_features["date_of_ultrasound_1"], test_features['sample_us_date'], test_features['manufacturer']]
                for item in test_cov:
                    assert len(item) == len(test_y)
                test_features = test_cov
            return train_X, train_y, train_features, test_X, test_y, test_features
        else:
            return train_X, train_y, test_X, test_y


    else:
        train_y = get_y(train_data)
        train_X = get_X(train_data, contrast, image_dim)

        test_y = get_y(test_data)
        test_X = get_X(test_data, contrast, image_dim)

        assert len(train_y) == len(train_X)
        assert len(test_y) == len(test_X)

        if get_features or get_cov:
            train_features = get_f(train_data)
            test_features = get_f(test_data)
            if get_cov:
                train_cov = [train_features["study_id"], train_features["age_at_baseline"], train_features["male"],
                             train_features["saggital"], train_features["sample_num"], train_features['kidney_side'],
                             train_features["date_of_ultrasound_1"], train_features['sample_us_date'], train_features['manufacturer']]
                test_cov = [test_features["study_id"], test_features["age_at_baseline"], test_features["male"],
                            test_features["saggital"], test_features["sample_num"], test_features['kidney_side'],
                            test_features["date_of_ultrasound_1"], test_features['sample_us_date'], test_features['manufacturer']]
                for item in train_cov:
                    assert len(item) == len(train_y)
                for item in test_cov:
                    assert len(item) == len(test_y)
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
        return get_sag(data, sort_by_date, split, contrast, image_dim, get_features, get_cov)
    elif views_to_get == "trans":
        return get_trans(data, sort_by_date, split, contrast, image_dim, get_features, get_cov)
    elif views_to_get == "siamese":
        return get_siamese(data, sort_by_date, split, contrast, image_dim, get_features, get_cov)



# =============================================================================
# def view_images(imgs, num_images_to_view=20, views_to_get="siamese"):
#     counter = 0
#     if views_to_get=="siamese":
#         for img in imgs:
#             if counter >= num_images_to_view:
#                 break
#             plt.figure()
#             plt.subplot(1, 2, 1)
#             plt.imshow(img[0], cmap='gray')
#             plt.subplot(1, 2, 2)
#             plt.imshow(img[1], cmap='gray')
#             plt.show()
#             counter += 1
#     else:
#         for img in imgs:
#             if counter >= num_images_to_view:
#                 break
#             plt.figure()
#             plt.subplot(1, 1, 1)
#             plt.imshow(img[0], cmap='gray')
#             counter += 1
# 
# =============================================================================
#datafile = "/home/lauren/preprocessed_images_20190402.pickle"
#train_X, train_y, f, test_X, test_y, x = load_dataset(views_to_get="siamese", pickle_file=datafile, get_cov=True)

#print(f)

# from sklearn.utils import shuffle
# train_X_shuffled = shuffle(train_X, random_state=42)
# train_X = train_X_shuffled[:int(len(train_X_shuffled)*0.8)]
# val_X = train_X_shuffled[int(len(train_X_shuffled)*0.8):]
#
# import os
# os.makedirs("ILSVRC2012_img_train")
# f = open("ILSVRC2012_img_train/ilsvrc12_train.txt", 'w')
# import scipy.misc
# counter = 0
# for img in train_X:
#     scipy.misc.imsave("ILSVRC2012_img_train/" + str(counter) +'.jpg', img[0])
#     f.write(str(counter) +'.jpg\n')
#     counter += 1
#     scipy.misc.imsave("ILSVRC2012_img_train/" + str(counter) + '.jpg', img[1])
#     f.write(str(counter) + '.jpg\n')
#     counter += 1
# f.close()
#
# os.makedirs("ILSVRC2012_img_val")
# f = open("ILSVRC2012_img_val/ilsvrc12_val.txt", 'w')
# counter = 0
# for img in val_X:
#     scipy.misc.imsave("ILSVRC2012_img_val/" + str(counter) + '.jpg', img[0])
#     f.write(str(counter) + '.jpg\n')
#     counter += 1
#     scipy.misc.imsave("ILSVRC2012_img_val/" + str(counter) + '.jpg', img[1])
#     f.write(str(counter) + '.jpg\n')
#     counter += 1
# f.close()
# #
#view_images(test_X)
