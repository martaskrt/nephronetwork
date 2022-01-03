import pandas as pd
import math
from skimage import exposure
import numpy as np
import argparse
import os
from PIL import Image
import json
#import ast
#from numpy import genfromtxt
#import matplotlib.pyplot as plt


def make_cov_labels(cov):
    cov_id_str = []
    num_samples = len(cov[0])
    # 0: study_id, 1: age_at_baseline, 2: gender (0 if male), 3: view (0 if saggital)...skip), 4: sample_num,
    # 5: kidney side, 6: date_of_US_1, 7: date of curr US, 8: manufacturer, 9: etiology
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
                if cov[j][i] == 0:
                    curr_sample.append("R")
                elif cov[j][i] == 1:
                    curr_sample.append("O")
            else:
                curr_sample.append(cov[j][i])

        cov_id = ""
        for item in curr_sample:
            cov_id += str(item) + "_"
        cov_id_str.append(cov_id[:-1])
    return cov_id_str


def open_file(file):
    #print(file)
    data = pd.read_pickle(file)
    #data = pd.read_pickle("preprocessed_images.pickle")
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


def get_f(data, samples_to_exclude=None, siamese=False,git_dir="/home/lauren"):

    study_id_date_map = pd.read_csv("C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/samples_with_studyids_and_usdates.csv")
    # "/Volumes/terminator/nephronetwork/preprocess/"

    #study_id_date_map = pd.read_csv("/Volumes/terminator/nephronetwork/preprocess/samples_with_studyids_and_usdates.csv")

    # study_id_date_map = pd.read_csv("/home/lauren/preprocess/samples_with_studyids_and_usdates.csv")
    features = dict()
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


def load_train_test_sets(data, sort_by_date, split, bottom_cut,contrast, image_dim, get_features, get_cov=False,
                         siamese=False, git_dir="/home/lauren"):
    patient_ids_and_ultrasound_dates = list(set(zip(data.study_id, data.date_of_ultrasound_1)))
    if sort_by_date:
        patient_ids_sorted = sorted(patient_ids_and_ultrasound_dates, key=lambda x: x[1])
    else:
        patient_ids_sorted = sorted(patient_ids_and_ultrasound_dates)
    train_ids, test_ids = train_test_split(patient_ids_sorted, split, bottom_cut)
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
            train_features = get_f(train_groups, samples_to_exclude=samples_to_exclude, siamese=True,git_dir = git_dir)
            if get_cov:
                train_cov = [train_features["study_id"], train_features["age_at_baseline"], train_features["male"],
                             train_features["saggital"], train_features["sample_num"], train_features['kidney_side'],
                             train_features["date_of_ultrasound_1"], train_features['sample_us_date'], train_features['manufacturer'],
                             ]
                for item in train_cov:
                    assert len(item) == len(train_y)
                train_features = make_cov_labels(train_cov)

        if split < 1:
            test_grouped = test_data.groupby(['study_id', 'sample_num', 'kidney_side'])
            test_groups = []
            for name, group in test_grouped:
                test_groups.append(group)
            test_X, samples_to_exclude = get_X(test_groups, contrast, image_dim, siamese=True)
            test_y = get_y(test_groups, siamese=True, samples_to_exclude=samples_to_exclude)
            assert len(test_y) == len(test_X)

            if get_features or get_cov:
                test_features = get_f(test_groups, samples_to_exclude=samples_to_exclude, siamese=True, git_dir = git_dir)
                if get_cov:
                    test_cov = [test_features["study_id"], test_features["age_at_baseline"], test_features["male"],
                                test_features["saggital"], test_features["sample_num"], test_features['kidney_side'],
                                test_features["date_of_ultrasound_1"], test_features['sample_us_date'], test_features['manufacturer']]

                    for item in test_cov:
                        assert len(item) == len(test_y)

                    test_features = make_cov_labels(test_cov)
                return train_X, train_y, train_features, test_X, test_y, test_features
            else:
                return train_X, train_y, test_X, test_y

        else:
            return train_X, train_y, train_features

    else:
        train_y = get_y(train_data)
        train_X = get_X(train_data, contrast, image_dim)

        test_y = get_y(test_data)
        test_X = get_X(test_data, contrast, image_dim)

        assert len(train_y) == len(train_X)
        assert len(test_y) == len(test_X)

        if get_features or get_cov:
            train_features = get_f(train_data,git_dir=git_dir)
            test_features = get_f(test_data,git_dir=git_dir)
            if get_cov:
                train_cov = [train_features["study_id"], train_features["age_at_baseline"], train_features["male"],
                             train_features["saggital"], train_features["sample_num"], train_features['kidney_side'],
                             train_features["date_of_ultrasound_1"], train_features['sample_us_date'], train_features['manufacturer'],
                             ]
                test_cov = [test_features["study_id"], test_features["age_at_baseline"], test_features["male"],
                            test_features["saggital"], test_features["sample_num"], test_features['kidney_side'],
                            test_features["date_of_ultrasound_1"], test_features['sample_us_date'], test_features['manufacturer'],
                            ]
                for item in train_cov:
                    assert len(item) == len(train_y)
                for item in test_cov:
                    assert len(item) == len(test_y)
                train_features = make_cov_labels(train_cov)
                test_features = make_cov_labels(test_cov)
            return train_X, train_y, train_features, test_X, test_y, test_features

        else:
            return train_X, train_y, test_X, test_y


def get_sag(data, sort_by_date, split, bottom_cut, contrast, image_dim, get_features, get_cov=False,git_dir="/home/lauren"):
    data = data[data.kidney_view == "Sag"]
    return load_train_test_sets(data, sort_by_date, split, bottom_cut, contrast, image_dim, get_features, get_cov=get_cov,git_dir=git_dir)


def get_trans(data, sort_by_date, split, bottom_cut, contrast, image_dim, get_features, get_cov=False,git_dir="/home/lauren"):
    data = data[data.kidney_view == "Trans"]
    return load_train_test_sets(data, sort_by_date, split, bottom_cut, contrast, image_dim, get_features, get_cov=get_cov,git_dir=git_dir)


def get_siamese(data, sort_by_date, split, bottom_cut, contrast, image_dim, get_features, get_cov=False, git_dir="/home/lauren"):
    return load_train_test_sets(data, sort_by_date, split, bottom_cut, contrast, image_dim, get_features,
                                get_cov=get_cov, siamese=True,git_dir=git_dir)


def load_dataset(split=0.7, sort_by_date=True, contrast=0, drop_bilateral=True, crop=0, get_features=False,
                 image_dim=256, views_to_get="all", get_cov=False, pickle_file="", bottom_cut=0, etiology="B",
                 git_dir="/home/lauren/"):
    data = open_file(pickle_file)

    if drop_bilateral:
        data = data[((data['hydro_kidney'] == data['kidney_side']) & (data['laterality'] == "Bilateral"))
                    | (data['laterality'] != "Bilateral")]
    data = data[data.crop_style == float(crop)]

    if etiology == 'O':
        data = data[data.etiology == int(1)]
    elif etiology == 'R':
        data = data[data.etiology == int(0)]

    if views_to_get == "sag":
        return get_sag(data, sort_by_date, split, bottom_cut, contrast, image_dim, get_features, get_cov, git_dir=git_dir)
    elif views_to_get == "trans":
        return get_trans(data, sort_by_date, split, bottom_cut, contrast, image_dim, get_features, get_cov, git_dir=git_dir)
    elif views_to_get == "siamese":
        return get_siamese(data, sort_by_date, split, bottom_cut, contrast, image_dim, get_features, get_cov, git_dir=git_dir)


def containsAll(str, set):
    """
    Check whether sequence str contains ALL of the items in set.
    Solution from: https://www.oreilly.com/library/view/python-cookbook/0596001673/ch03s07.html
    """
    return 0 not in [c in str for c in set]


def extract_st_us_num(file_string):
    split1 = file_string.lower().split("-")[0]
    split2 = split1.split(" ")[len(split1.split(" ")) - 1][:-1]

    return split2


def get_view(filename):
    if containsAll(filename.lower(), "trv"):
        view = "trv"
    elif containsAll(filename.lower(), "sag"):
        view = "sag"
    else:
        view = "unknown"

    return view


def get_side(filename):
    side = filename.upper()[-18]

    if side == "R":
        side = "Right"
    elif side == "L":
        side = "Left"
    else:
        side = "NA"

    return side

## read in images
def get_st_files(study_id, img_dir, data_sheet):
    out_dict = {"Right": dict(), "Left": dict()}

    print("Img dir: " + img_dir)

    for file in os.listdir(img_dir):

        print("File:" + file)

        if containsAll(file, "-preprocced.png"):

            num = extract_st_us_num(file)
            view = get_view(filename=file)
            side = get_side(filename=file)
            print("US number: " + num)
            out_dict["Sex"] = list(data_sheet.query('ID == ' + str(study_id)).Sex)[0]
            out_dict["BL_date"] = list(data_sheet.query('ID == ' + str(study_id)).DOB)[0]

            # print("Opening: " + file)
            try:
                # out_dict[side][num][view] = np.array(Image.open(os.path.join(img_dir, file)).convert('L')).tolist()
                out_dict[side][num][view] = os.path.join(img_dir, file)
            except KeyError:
                out_dict[side][num] = dict()
                # out_dict[side][num][view] = np.array(Image.open(os.path.join(img_dir, file)).convert('L')).tolist()
                out_dict[side][num][view] = os.path.join(img_dir, file)
            except FileNotFoundError:
                out_dict[side][num][view] = "NA"

            if len(list(data_sheet.query('ID == ' + str(study_id) + ' & view_side == "' + side + '"').surg)) > 0:
                out_dict[side]["surgery"] = \
                    list(data_sheet.query('ID == ' + str(study_id) + ' & view_side == "' + side + '"').surg)[0]
                try:
                    out_dict[side][num]["US_machine"] = \
                        str(list(data_sheet.query('ID == ' + str(study_id) +
                                                  ' & view_side == "' + side + '"').US_machine)[0]) + "ST"
                    out_dict[side][num]["SFU"] = \
                        list(data_sheet.query('ID == ' + str(study_id) +
                                                  ' & view_side == "' + side + '"').SFU)[0]
                    out_dict[side][num]["Age_wks"] = \
                        list(data_sheet.query('ID == ' + str(study_id) +
                                                  ' & view_side == "' + side + '"').age_at_US_wk)[0]
                    out_dict[side][num]["ApD"] = \
                        list(data_sheet.query('ID == ' + str(study_id) +
                                                  ' & view_side == "' + side + '"').ApD)[0]


                except KeyError:
                    out_dict[side][num] = dict()
                    out_dict[side][num]["US_machine"] = \
                        str(list(data_sheet.query('ID == ' + str(study_id) +
                                                  ' & view_side == "' + side + '"').US_machine)[0]) + "ST"
                    out_dict[side][num]["SFU"] = \
                        list(data_sheet.query('ID == ' + str(study_id) +
                                                  ' & view_side == "' + side + '"').SFU)[0]
                    out_dict[side][num]["Age_wks"] = \
                        list(data_sheet.query('ID == ' + str(study_id) +
                                                  ' & view_side == "' + side + '"').age_at_US_wk)[0]
                    out_dict[side][num]["ApD"] = \
                        list(data_sheet.query('ID == ' + str(study_id) +
                                                  ' & view_side == "' + side + '"').ApD)[0]

            else:
                out_dict[side]["surgery"] = "NA"
                out_dict[side][num]["US_machine"] = "NA"
                out_dict[side][num]["Age_wks"] = "NA"
                out_dict[side][num]["ApD"] = "NA"
                out_dict[side][num]["SFU"] = "NA"

    return out_dict


def load_st_data(data_folder="C:/Users/lauren erdman/Desktop/kidney_img/HN/silent_trial/ImageOutput/HN Outputs", ## put these in the args
                 data_sheet="C:/Users/lauren erdman/Desktop/kidney_img/HN/silent_trial/SilentTrial_Datasheet.csv"):
    my_dat = pd.read_csv(data_sheet)

    img_dict = dict()
    for i in my_dat["ID"].unique():
        study_folder = data_folder + "/Study ID " + str(i) + "/"
        if os.path.exists(study_folder):
            print("Study ID " + str(i))
            img_dict["STID"+str(i)] = get_st_files(study_id=i, img_dir=study_folder, data_sheet=my_dat)

    return img_dict

### CREATE SECOND FUNCTION SIMILAR TO combine_st_orig WHICH COMBINES DATA IN DA DICTIONARY
            #       (VARIABLE FOR WHAT THE SOURCES ARE)
def combine_st_orig(in_dict, X, y, feat, pt_info_file):

    pt_data = pd.read_csv(pt_info_file)

    for i in range(len(feat)):
        feat_split = feat[i].split("_")
        study_id = str(int(float(feat_split[0])))

        print("Study ID: " + str(study_id))

        side = feat_split[4]
        surg = y[i]
        sex = 1 if str(feat_split[2]) == "M" else 2
        us_num = feat_split[3]
        bl_date = feat_split[5]
        us_machine = feat_split[7]

        us_age = pt_data.query('US_num == '+str(us_num)+' & study_id == '+str(study_id)).US_age_wk
        us_age = "NA" if len(us_age) == 0 else list(us_age)[0]

        print("US age (wks): " + str(us_age))
        print("Surgery: " + str(surg))

        sag_out_file = "C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/Original_CNN_Imgs/ORIG" + \
                       study_id + "_" + side + "_" + us_num + "_sag.jpg"
        trv_out_file = "C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/Original_CNN_Imgs/ORIG" + \
                       study_id + "_" + side + "_" + us_num + "_trv.jpg"

        Image.fromarray(X[i, 0, :, :]).convert('L').save(sag_out_file)
        Image.fromarray(X[i, 1, :, :]).convert('L').save(trv_out_file)

        try:
            in_dict["ORIG" + study_id]["Sex"] = sex
            in_dict["ORIG" + study_id]["BL_date"] = bl_date
        except KeyError:
            in_dict["ORIG" + study_id] = dict()
            in_dict["ORIG" + study_id]["Sex"] = sex
            in_dict["ORIG" + study_id]["BL_date"] = bl_date

        try:
            in_dict["ORIG" + study_id][side]['surgery'] = surg
        except KeyError:
            in_dict["ORIG" + study_id][side] = dict()
            in_dict["ORIG" + study_id][side]['surgery'] = surg

        try:
            in_dict["ORIG" + study_id][side][us_num]['sag'] = sag_out_file
            in_dict["ORIG" + study_id][side][us_num]['trv'] = trv_out_file
            # in_dict["ORIG" + study_id][side][us_num]['sag'] = X[i, 0, :, :].tolist()
            # in_dict["ORIG" + study_id][side][us_num]['trv'] = X[i, 1, :, :].tolist()
            in_dict["ORIG" + study_id][side][us_num]['US_machine'] = us_machine
            in_dict["ORIG" + study_id][side][us_num]['Age_wks'] = us_age
        except KeyError:
            in_dict["ORIG" + study_id][side][us_num] = dict()
            in_dict["ORIG" + study_id][side][us_num]['sag'] = sag_out_file
            in_dict["ORIG" + study_id][side][us_num]['trv'] = trv_out_file
            # in_dict["ORIG" + study_id][side][us_num]['sag'] = X[i, 0, :, :].tolist()
            # in_dict["ORIG" + study_id][side][us_num]['trv'] = X[i, 1, :, :].tolist()
            in_dict["ORIG" + study_id][side][us_num]['US_machine'] = us_machine
            in_dict["ORIG" + study_id][side][us_num]['Age_wks'] = us_age

    return in_dict

def main():
    parser = argparse.ArgumentParser()
    ## old args:
    parser.add_argument('--epochs', default=50, type=int, help="Number of epochs")
    parser.add_argument('--batch_size', default=64, type=int, help="Batch size")
    parser.add_argument('--lr', default=0.001, type=float, help="Learning rate")
    parser.add_argument('--momentum', default=0.9, type=float, help="Momentum")
    parser.add_argument('--adam', action="store_true", help="Use Adam optimizer instead of SGD")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay")
    parser.add_argument("--num_workers", default=1, type=int, help="Number of CPU workers")


    parser.add_argument("--contrast", default=1, type=int, help="Image contrast to train on")
    parser.add_argument("--bottom_cut", default=0.0, type=float, help="proportion of dataset to cut from bottom")
    parser.add_argument("--etiology", default="B", help="O (obstruction), R (reflux), B (both)")
    parser.add_argument("--crop", default=0, type=int, help="Crop setting (0=big, 1=tight)")
    parser.add_argument("--gender", default=None, type=str, help="choose from 'male' and 'female'")

    ## my args:
    parser.add_argument('--split', default=0.7, type=float, help="Output dataset split")
    parser.add_argument("--git_dir", default="./", help="Directory to save model checkpoints to")
        ## update to preprocessed_images_20190617.pickle
    parser.add_argument("--datafile", default="C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_20190617.pickle")
    parser.add_argument("--st_datafile", default="C:/Users/lauren erdman/Desktop/kidney_img/HN/silent_trial/SilentTrial_Datasheet.csv",
                        help="Silent trial datasheet")
    parser.add_argument("--json_out", default="C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/preprocessed_images_SickKids_wST_filenames_20210216.json", help="Directory to save model checkpoints to")
    parser.add_argument("--orig_extra_dat", default="C:/Users/lauren erdman/Desktop/kidney_img/HN/SickKids/Orig_data_age_apd_sfu.csv", help="Directory to save model checkpoints to")


    args = parser.parse_args()

    print("ARGS" + '\t' + str(args))

    train_X, train_y, train_cov = load_dataset(views_to_get="siamese",
                                              sort_by_date=True,
                                              pickle_file=args.datafile,
                                              contrast=args.contrast,
                                              split=1,
                                              get_cov=True,
                                              bottom_cut=args.bottom_cut,
                                              etiology=args.etiology,
                                              crop=args.crop,
                                              git_dir=args.git_dir)
    print("original data loaded")

    st_data = load_st_data(data_sheet=args.st_datafile)
    print("silent trial data loaded")

    full_data = combine_st_orig(in_dict=st_data, X=train_X, y=train_y, feat=train_cov, pt_info_file=args.orig_extra_dat)
    print("data merged")

    with open(args.json_out, 'w') as fp:
        json.dump(full_data, fp)

    ## write final data to a json then read in as dict in dataloader in training


if __name__ == '__main__':
    main()



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
