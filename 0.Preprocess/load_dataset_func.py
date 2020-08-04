import pandas as pd
import math
from skimage import exposure
import numpy as np
#import ast
#from numpy import genfromtxt
#import matplotlib.pyplot as plt

def make_cov_labels(cov):
    cov_id_str = []
    num_samples = len(cov[0])
    # 0: study_id, 1: age_at_baseline, 2: gender (0 if male), 3: view (0 if saggital)...skip), 4: sample_num,
    # 5: kidney side, 6: date_of_US_1, 7: date of curr US, 8: manufacturer, 9: etiology, 10: vur_grade
    for i in range(num_samples):
        curr_sample = []
        for j in range(len(cov)):
            if j == 2:
                if cov[j][i] == 0:
                    curr_sample.append("M")
                elif cov[j][i] == 1:
                    curr_sample.append("F")
            elif j == 3:
                continue
            elif j == 4:
                curr_sample.append(int(cov[j][i]))
            elif j == 9:
                curr_sample.append("g" + str(cov[j][i]))
            else:
                curr_sample.append(cov[j][i])

        cov_id = ""
        for item in curr_sample:
            cov_id += str(item) + "_"
        cov_id_str.append(cov_id[:-1])
    return cov_id_str

def open_file(file):
    data = pd.read_pickle(file)
    return data


def train_test_split(patient_ids_sorted, split, bottom_cut):
    train_split = math.floor(len(patient_ids_sorted)*split)
    data_bottom_cut = math.floor(len(patient_ids_sorted)*bottom_cut)
    train_ids = list(map(lambda x: x[0], patient_ids_sorted[data_bottom_cut:train_split]))
    test_ids = list(map(lambda x: x[0], patient_ids_sorted[train_split:]))
    return train_ids, test_ids


def get_y(data, siamese=False, samples_to_exclude=None):
    labels = []
    if siamese:
        num_samples = len(data)
        for i in range(num_samples):
            if i in samples_to_exclude:
                continue
           
            #if data[i]['kidney_side'].iloc[0] == "Left" and data[i]['hydro_kidney'].iloc[0] == "Left":
            #    labels.append(int(data[i]['function'].values[0])/100)
            #elif data[i]['kidney_side'].iloc[0] == "Right" and data[i]['hydro_kidney'].iloc[0] == "Right":
            #    labels.append(int(100-data[i]['function'].values[0])/100)
            #elif data[i]['kidney_side'].iloc[0] == "Right" and data[i]['hydro_kidney'].iloc[0] == "Left":
            #    labels.append(int(data[i]['function'].values[0])/100)
            #elif data[i]['kidney_side'].iloc[0] == "Left" and data[i]['hydro_kidney'].iloc[0] == "Right":
            #    labels.append(int(100-data[i]['function'].values[0])/100)
            if data[i]['hydro_kidney'].iloc[0] == "Left":
                labels.append(int(data[i]['function'].values[0])/100)
            elif data[i]['hydro_kidney'].iloc[0] == "Right":
                labels.append(int(100-data[i]['function'].values[0])/100)

    return labels


def get_X(data, contrast, image_dim, siamese=False):

    num_samples = len(data)
    X = []
    samples_to_exclude = []
    if siamese:
        for i in range(num_samples):
            if (data[i]['image'].shape[0] < 4):
                samples_to_exclude.append(i)
                continue
            group = np.zeros((4, image_dim, image_dim))
            present = [0,0,0,0]
            for j in range(4):
                
                if data[i]['kidney_view'].iloc[j] == "Sag" and data[i]['kidney_side'].iloc[j] == "Left":
                    idx = 0
                elif data[i]['kidney_view'].iloc[j] == "Trans" and data[i]['kidney_side'].iloc[j] == "Left":
                    idx = 1
                elif data[i]['kidney_view'].iloc[j] == "Sag" and data[i]['kidney_side'].iloc[j] == "Right":
                    idx = 2
                elif data[i]['kidney_view'].iloc[j] == "Trans" and data[i]['kidney_side'].iloc[j] == "Right":
                    idx = 3
                present[idx] = 1
                image = data[i]['image'].iloc[j].reshape((image_dim,image_dim))
                if contrast == 0:
                    group[idx, :, :] = image
                elif contrast == 1:
                    group[idx, :, :] = exposure.equalize_hist(image)
                elif contrast == 2:
                    group[idx, :, :] = exposure.equalize_adapthist(image)
                elif contrast == 3:
                    group[idx, :, :] = exposure.rescale_intensity(image)
            if np.sum(present) == 4:            
                X.append(group)
            else:
                samples_to_exclude.append(i)
        X = np.array(X)
        return X, samples_to_exclude


def get_f(data, samples_to_exclude=None, siamese=False):

    study_id_date_map = pd.read_csv("/home/marta/nephronetwork-github/nephronetwork/0.Preprocess/samples_with_studyids_and_usdates.csv")
    features = {}
    if siamese:
        for column in data[0]:
            if column not in ['laterality', 'surgery', 'crop_style', 'hydro_kidney', 'image', 'kidney_view']:
                features[column] = []
    features["saggital"] = []
    features["male"] = []
    features["sample_us_date"] = []


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
    return features


def load_train_test_sets(data, sort_by_date, split, bottom_cut,contrast, image_dim, get_features, get_cov=False, siamese=False):
    patient_ids_and_ultrasound_dates = list(set(zip(data.study_id, data.date_of_ultrasound_1)))
    if sort_by_date:
        patient_ids_sorted = sorted(patient_ids_and_ultrasound_dates, key=lambda x: x[1])
    else:
        patient_ids_sorted = sorted(patient_ids_and_ultrasound_dates)
    train_ids, test_ids = train_test_split(patient_ids_sorted, split, bottom_cut)
    train_data = data[data.study_id.isin(train_ids)]
    test_data = data[data.study_id.isin(test_ids)]
    if siamese:
        train_grouped = train_data.groupby(['study_id', 'sample_num'])
        train_groups = []
        
        for name, group in train_grouped:
            if len(group) < 4:
                continue
            train_groups.append(group)
 
        train_X, samples_to_exclude = get_X(train_groups, contrast, image_dim, siamese=True)
        train_y = get_y(train_groups, siamese=True, samples_to_exclude=samples_to_exclude)
        assert len(train_y) == len(train_X)
        if get_features or get_cov:
            train_features = get_f(train_groups, samples_to_exclude=samples_to_exclude, siamese=True)
            if get_cov:
                train_cov = [train_features["study_id"], train_features["age_at_baseline"], train_features["male"],
                             train_features["saggital"], train_features["sample_num"], train_features['kidney_side'],
                             train_features["date_of_ultrasound_1"], train_features['sample_us_date'], train_features['manufacturer'],
                             train_features["scan_type"],]
                counter = 0
                train_features = make_cov_labels(train_cov)
        test_grouped = test_data.groupby(['study_id', 'sample_num'])
        test_groups = []
        for name, group in test_grouped:
            if len(group) < 4:
                continue
            test_groups.append(group)
        test_X, samples_to_exclude = get_X(test_groups, contrast, image_dim, siamese=True)
        test_y = get_y(test_groups, siamese=True, samples_to_exclude=samples_to_exclude)
        assert len(test_y) == len(test_X)

        if get_features or get_cov:
            test_features = get_f(test_groups, samples_to_exclude=samples_to_exclude, siamese=True)
            if get_cov:
                test_cov = [test_features["study_id"], test_features["age_at_baseline"], test_features["male"],
                            test_features["saggital"], test_features["sample_num"], test_features['kidney_side'],
                            test_features["date_of_ultrasound_1"], test_features['sample_us_date'], test_features['manufacturer'],
                            test_features["scan_type"],]

                for item in test_cov:
                    assert len(item) == len(test_y)

                test_features = make_cov_labels(test_cov)
            return train_X, train_y, train_features, test_X, test_y, test_features
        else:
            return train_X, train_y, test_X, test_y





def load_dataset(split=0.7, sort_by_date=True, contrast=0, drop_bilateral=True, crop=0, get_features=False,
                 image_dim=300, views_to_get="all", get_cov=False, pickle_file="", bottom_cut=0, etiology="B", hydro_only=False, gender=None,
                 high_vur=False):

    data = open_file(pickle_file)
    print("Loading FUNC data............")
    if drop_bilateral:
        data = data[((data['hydro_kidney'] == data['kidney_side']) & (data['laterality'] == "Bilateral"))
                    | (data['laterality'] != "Bilateral")]
    if hydro_only:
        data = data[(data['hydro_kidney'] == data['kidney_side'])]
    data = data[data.crop_style == float(crop)]
    if gender == 'male':
        data = data[data.gender == "Male"]
    elif gender == 'female':
        data = data[data.gender == "Female"]
        

    return load_train_test_sets(data, sort_by_date, split, bottom_cut, contrast, image_dim, get_features, get_cov=get_cov, siamese=True)





