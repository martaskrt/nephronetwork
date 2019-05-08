# -*- coding: utf-8 -*-
"""
Created on Tue May  7 13:22:30 2019

@author: larun
"""

##
##      LOADING NECESSARY LIBRARIES 
##

import os
import argparse
import torch 

##
##      FUNCTIONS USED IN MAIN FUNCTION
##

# LOAD .pth FILES TO ANALYZE
def load_pth_files_names(pth_folder):
    pth_files = []
    for subdir, dirs, files in os.walk(pth_folder):
        for file in files:            
            pth_files.append(os.path.join(subdir,file))
    return pth_files


## OPEN EACH OF THE PTHS -- they're each a dictionary with 1/5 of the data you'll want to use
def open_file(file):
    print("Opening" + file)
    data = torch.load(file)
    #print(data.head)
    return data

## GET FOLD NUMBER FROM FILE NAME
def get_fold(file_name):
    return file_name.split("_")[1]

## CREAT PANDAS DATAFRAME FOR EACH FOLD
def make_subset_df(my_fold_data,fold):
    
    # 0: study_id, 1: age_at_baseline, 2: gender (0 if male), 3: view (0 if saggital)...skip), 
    # 4: sample_num, 5: kidney side, 6: date_of_US_1, 7: date of curr US
    train_df = {'full_ID': my_fold_data['patient_ID_train'],
                'Fold': [fold]*len(my_fold_data['patient_ID_train']),
                'Target': my_fold_data['all_targets_train'].tolist(),
                'Pred_val': my_fold_data['all_pred_prob_train'].tolist(),
                'study_id': [x.split("_")[0] for x in my_fold_data['patient_ID_train']],
                'age_at_baseline': [x.split("_")[1] for x in my_fold_data['patient_ID_train']],
                'gender': [x.split("_")[2] for x in my_fold_data['patient_ID_train']],
                'us_num': [x.split("_")[3] for x in my_fold_data['patient_ID_train']],
                'kidney_side': [x.split("_")[4] for x in my_fold_data['patient_ID_train']],
                'date_of_us1': [x.split("_")[5] for x in my_fold_data['patient_ID_train']],
                'date_of_current_us': [x.split("_")[6] for x in my_fold_data['patient_ID_train']],
                'us_man': [x.split("_")[7] for x in my_fold_data['patient_ID_train']]}
    
    val_df = {'full_ID': my_fold_data['patient_ID_val'],
                'Fold': [fold]*len(my_fold_data['patient_ID_val']),
                'Target': my_fold_data['all_targets_val'].tolist(),
                'Pred_val': my_fold_data['all_pred_prob_val'].tolist(),
                'study_id': [x.split("_")[0] for x in my_fold_data['patient_ID_val']],
                'age_at_baseline': [x.split("_")[1] for x in my_fold_data['patient_ID_val']],
                'gender': [x.split("_")[2] for x in my_fold_data['patient_ID_val']],
                'us_num': [x.split("_")[3] for x in my_fold_data['patient_ID_val']],
                'kidney_side': [x.split("_")[4] for x in my_fold_data['patient_ID_val']],
                'date_of_us1': [x.split("_")[5] for x in my_fold_data['patient_ID_val']],
                'date_of_current_us': [x.split("_")[6] for x in my_fold_data['patient_ID_val']],
                'us_man': [x.split("_")[7] for x in my_fold_data['patient_ID_val']]}

    test_df = {'full_ID': my_fold_data['patient_ID_test'],
                'Fold': [fold]*len(my_fold_data['patient_ID_test']),
                'Target': my_fold_data['all_targets_test'].tolist(),
                'Pred_val': my_fold_data['all_pred_prob_test'].tolist(),
                'study_id': [x.split("_")[0] for x in my_fold_data['patient_ID_test']],
                'age_at_baseline': [x.split("_")[1] for x in my_fold_data['patient_ID_test']],
                'gender': [x.split("_")[2] for x in my_fold_data['patient_ID_test']],
                'us_num': [x.split("_")[3] for x in my_fold_data['patient_ID_test']],
                'kidney_side': [x.split("_")[4] for x in my_fold_data['patient_ID_test']],
                'date_of_us1': [x.split("_")[5] for x in my_fold_data['patient_ID_test']],
                'date_of_current_us': [x.split("_")[6] for x in my_fold_data['patient_ID_test']],
                'us_man': [x.split("_")[7] for x in my_fold_data['patient_ID_test']]}

    return train_df,val_df,test_df

def make_full_df(pth_folder):
    my_pth_files = load_pth_files_names(pth_folder)
    
    for file_name in my_pth_files: 
        fold_data = open_file(pth_folder + file_name)
        fold_num = get_fold(file_name)
        if file_name == my_pth_files[0]:
            my_train_df, my_val_df, my_test_df = make_subset_df(fold_data,fold_num)
        else:
            new_train,new_val,new_test = make_subset_df(fold_data,fold_num)
            
            my_train_df.append(new_train)
            my_val_df.append(new_val)
            my_test_df.append(new_test)
            
    return my_train_df, my_val_df, my_test_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_folder', default=50, type=int, help="Number of epochs")

    args = parser.parse_args()

    train,val,test = make_full_df(args.checkpoint_folder)
    
    file_root = args.checkpoint_folder.split("/")[len(args.checkpoint_folder.split("/")) - 1]
    
    train.to_csv(args.checkpoint_folder + file_root + "_train.csv")
    val.to_csv(args.checkpoint_folder + file_root + "_val.csv")
    test.to_csv(args.checkpoint_folder + file_root + "_test.csv")



if __name__ == '__main__':
    main()
