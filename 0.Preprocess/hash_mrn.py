
"""
Script to generate hash for MRN code.
Flow:
h = Hashed_file_name --> hash2rank[h] = p (position in original list) --> rank2mrn[p] = original mrn code
"""

import numpy as np
import os
import pickle
import csv

np.random.seed(9)


mrn2rank = {}
rank2mrn = {}

hash2rank = {}
rank2hash = {}

hash2mrn = {}
mrn2hash = {}

shuffled_rank_array = []

rootdir = '/home/martaskreta/Desktop/CSC2516/all_kidney_images/'
rootdir_hashed = '/home/martaskreta/Desktop/CSC2516/all_kidney_images_hashed/'


def hash_mrn(mrn_codes):
    # position of mrn in ordered list
    rank = 0
    for code in mrn_codes:
        mrn2rank[code] = rank
        rank2mrn[rank] = code
        rank += 1

    # shuffle rank array with random seed 9
    global shuffled_rank_array
    shuffled_rank_array = np.random.permutation(rank)

    # map shuffled rank to rank in original list
    for i in range(len(shuffled_rank_array)):
        hash2rank[shuffled_rank_array[i]] = i
        rank2hash[i] = shuffled_rank_array[i]

    # map shuffled rank to original mrn
    for i in range(len(shuffled_rank_array)):
        hash2mrn[rank2hash[i]] = rank2mrn[i]
        mrn2hash[rank2mrn[i]] = rank2hash[i]

    output_dict = {
        "hash2mrn": hash2mrn,
        "mrn2hash": mrn2hash
    }

    pickle_out = open("hashed_mrns.pickle",'wb')
    pickle.dump(output_dict, pickle_out)
    pickle_out.close()

    with open("hashed_mrns.csv", mode='w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['file_name', 'original_mrn'])
        for key in hash2mrn:
            writer.writerow([key, hash2mrn[key]])


    return hash2mrn, mrn2hash

def get_mrn_codes(rootdir):
    all_folders = set()
    for subdir, dirs, files in os.walk(rootdir):
        dir_name = subdir.split("/")[-1]
        if dir_name != "":
            rootname = dir_name[1:].split("_")[0]
            all_folders.add(rootname)


    all_folders = sorted(all_folders)
    return all_folders

def rename_folders(mrn2hash):
    for subdir, dirs, files in os.walk(rootdir_hashed):
        dir_name = subdir.split("/")[-1]
        dir_path = "/".join(subdir.split("/")[:-1])
        if dir_name != "":
            rootname = dir_name[1:].split("_")[0]
            patient_num = dir_name[1:].split("_")[1]
            hashed_name = mrn2hash[rootname]

            new_file_name = str(hashed_name) + "_" + str(patient_num)
            new_file_name = dir_path + "/" + new_file_name
            old_file_name = dir_path + "/" + dir_name

            os.rename(old_file_name, new_file_name)


mrn_codes = get_mrn_codes(rootdir)
hash2mrn, mrn2hash = hash_mrn(mrn_codes)
rename_folders(mrn2hash)
