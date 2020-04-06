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
import pandas as pd

##
##      FUNCTIONS USED IN MAIN FUNCTION
##

# LOAD .pth FILES TO ANALYZE
def load_pth_files_names(pth_folder):
    pth_files = []
    for subdir, dirs, files in os.walk(pth_folder):
        for file in files:            
            if file.lower()[-4:] == '.pth':
                pth_files.append(os.path.join(subdir,file))
    print(pth_files)
    return pth_files

## OPEN EACH OF THE PTHS -- they're each a dictionary with 1/5 of the data you'll want to use
def open_file(file):
    print("Opening" + file)
    data = torch.load(file, map_location=torch.device("cpu"))
    #print(data.head)
    return data

## GET FOLD NUMBER FROM FILE NAME
def get_fold(file_name,fold_in):
    pth_file = file_name.split("/")[len(file_name.split("/")) - 1]
    fold = pth_file.split("_")[fold_in]
    return fold

## CREAT PANDAS DATAFRAME FOR EACH FOLD
def make_subset_wfold_df(my_fold_data,fold):
    
    # 0: study_id, 1: age_at_baseline, 2: gender (0 if male), 3: view (0 if saggital)...skip), 
    # 4: sample_num, 5: kidney side, 6: date_of_US_1, 7: date of curr US
    # print("ID splits: " +  ' '.join(my_fold_data['patient_ID_train'][1].split("_")))
    
    print("Number of train samples: " + str(len(my_fold_data['patient_ID_train'])))
    train_dict = {'full_ID': my_fold_data['patient_ID_train'],
                'Fold': [fold]*len(my_fold_data['patient_ID_train']),
                'Target': my_fold_data['all_targets_train'].tolist(),
                'Pred_val': my_fold_data['all_pred_prob_train'].tolist(),
                'study_id': [x.split("_")[0] for x in my_fold_data['patient_ID_train']],
                'age_at_baseline': [x.split("_")[1] for x in my_fold_data['patient_ID_train']],
                'gender': [x.split("_")[2] for x in my_fold_data['patient_ID_train']],
                'us_num': [x.split("_")[3] for x in my_fold_data['patient_ID_train']],
                'kidney_side': [x.split("_")[4] for x in my_fold_data['patient_ID_train']],
                'date_of_us1': [x.split("_")[5] for x in my_fold_data['patient_ID_train']],
                'date_of_current_us': [x.split("_")[6] for x in my_fold_data['patient_ID_train']]}#,
                #'us_man': [x.split("_")[7] for x in my_fold_data['patient_ID_train']]}
    
    print("Number of val samples: " + str(len(my_fold_data['patient_ID_val'])))
    val_dict = {'full_ID': my_fold_data['patient_ID_val'],
                'Fold': [fold]*len(my_fold_data['patient_ID_val']),
                'Target': my_fold_data['all_targets_val'].tolist(),
                'Pred_val': my_fold_data['all_pred_prob_val'].tolist(),
                'study_id': [x.split("_")[0] for x in my_fold_data['patient_ID_val']],
                'age_at_baseline': [x.split("_")[1] for x in my_fold_data['patient_ID_val']],
                'gender': [x.split("_")[2] for x in my_fold_data['patient_ID_val']],
                'us_num': [x.split("_")[3] for x in my_fold_data['patient_ID_val']],
                'kidney_side': [x.split("_")[4] for x in my_fold_data['patient_ID_val']],
                'date_of_us1': [x.split("_")[5] for x in my_fold_data['patient_ID_val']],
                'date_of_current_us': [x.split("_")[6] for x in my_fold_data['patient_ID_val']]}#,
                #'us_man': [x.split("_")[7] for x in my_fold_data['patient_ID_val']]}

    print("Number of test samples: " + str(len(my_fold_data['patient_ID_test'])))
    test_dict = {'full_ID': my_fold_data['patient_ID_test'],
                'Fold': [fold]*len(my_fold_data['patient_ID_test']),
                'Target': my_fold_data['all_targets_test'].tolist(),
                'Pred_val': my_fold_data['all_pred_prob_test'].tolist(),
                'study_id': [x.split("_")[0] for x in my_fold_data['patient_ID_test']],
                'age_at_baseline': [x.split("_")[1] for x in my_fold_data['patient_ID_test']],
                'gender': [x.split("_")[2] for x in my_fold_data['patient_ID_test']],
                'us_num': [x.split("_")[3] for x in my_fold_data['patient_ID_test']],
                'kidney_side': [x.split("_")[4] for x in my_fold_data['patient_ID_test']],
                'date_of_us1': [x.split("_")[5] for x in my_fold_data['patient_ID_test']],
                'date_of_current_us': [x.split("_")[6] for x in my_fold_data['patient_ID_test']]}#,
                #'us_man': [x.split("_")[7] for x in my_fold_data['patient_ID_test']]}

    train_df = pd.DataFrame.from_dict(train_dict)
    val_df = pd.DataFrame.from_dict(val_dict)
    test_df = pd.DataFrame.from_dict(test_dict)

    return train_df,val_df,test_df

## CREAT PANDAS DATAFRAME FOR EACH FOLD
def make_subset_nofold_df(my_fold_data):
    
    # 0: study_id, 1: age_at_baseline, 2: gender (0 if male), 3: view (0 if saggital)...skip), 
    # 4: sample_num, 5: kidney side, 6: date_of_US_1, 7: date of curr US
    # print("ID splits: " +  ' '.join(my_fold_data['patient_ID_train'][1].split("_")))
    
    print("Number of train samples: " + str(len(my_fold_data['patient_ID_train'])))
    train_dict = {'full_ID': my_fold_data['patient_ID_train'],
                'Target': my_fold_data['all_targets_train'].tolist(),
                'Pred_val': my_fold_data['all_pred_prob_train'].tolist(),
                'study_id': [x.split("_")[0] for x in my_fold_data['patient_ID_train']],
                'age_at_baseline': [x.split("_")[1] for x in my_fold_data['patient_ID_train']],
                'gender': [x.split("_")[2] for x in my_fold_data['patient_ID_train']],
                'us_num': [x.split("_")[3] for x in my_fold_data['patient_ID_train']],
                'kidney_side': [x.split("_")[4] for x in my_fold_data['patient_ID_train']],
                'date_of_us1': [x.split("_")[5] for x in my_fold_data['patient_ID_train']],
                'date_of_current_us': [x.split("_")[6] for x in my_fold_data['patient_ID_train']]}#,
                #'us_man': [x.split("_")[7] for x in my_fold_data['patient_ID_train']]}
 
    print("Number of test samples: " + str(len(my_fold_data['patient_ID_test'])))
    test_dict = {'full_ID': my_fold_data['patient_ID_test'],
                'Target': my_fold_data['all_targets_test'].tolist(),
                'Pred_val': my_fold_data['all_pred_prob_test'].tolist(),
                'study_id': [x.split("_")[0] for x in my_fold_data['patient_ID_test']],
                'age_at_baseline': [x.split("_")[1] for x in my_fold_data['patient_ID_test']],
                'gender': [x.split("_")[2] for x in my_fold_data['patient_ID_test']],
                'us_num': [x.split("_")[3] for x in my_fold_data['patient_ID_test']],
                'kidney_side': [x.split("_")[4] for x in my_fold_data['patient_ID_test']],
                'date_of_us1': [x.split("_")[5] for x in my_fold_data['patient_ID_test']],
                'date_of_current_us': [x.split("_")[6] for x in my_fold_data['patient_ID_test']]}#,
                #'us_man': [x.split("_")[7] for x in my_fold_data['patient_ID_test']]}

    train_df = pd.DataFrame.from_dict(train_dict)
    test_df = pd.DataFrame.from_dict(test_dict)

    return train_df,test_df
    
def make_full_df(pth_folder,cv,fold_in=0):
    my_pth_files = load_pth_files_names(pth_folder)
    
    if cv:     
        for file_name in my_pth_files: 
        
            fold_data = open_file(file_name)
            fold_num = get_fold(file_name, fold_in)
            if file_name == my_pth_files[0]:
                my_train_df, my_val_df, my_test_df = make_subset_wfold_df(fold_data,fold_num)
                print(file_name + " pandas dataframes created.")
            else:
                new_train,new_val,new_test = make_subset_wfold_df(fold_data,fold_num)
                print(file_name + " pandas dataframes created.")
                #print(new_train.shape)
                my_train_df = my_train_df.append(new_train)
                my_val_df = my_val_df.append(new_val)
                my_test_df = my_test_df.append(new_test)
        print("Final training dataframe shape: \n")
        return my_train_df, my_val_df, my_test_df
    else: 
        assert len(my_pth_files) == 1
        mydata = open_file(my_pth_files[0])
        my_train_df, my_test_df = make_subset_nofold_df(mydata)
        print(my_pth_files[0] + " pandas dataframes created.")
                
        print("Final training dataframe shape: \n")
    # print(my_train_df.shape)            
        return my_train_df, my_test_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_folder', help="Number of epochs")
    parser.add_argument('--cv', action='store_true',default=False, help="Read in CV")
    parser.add_argument('--output_file', type=str, default='', help="Output file name")
    parser.add_argument('--fold_inx', type=int, default=0, help="Position the fold number is found in")

    args = parser.parse_args()

    if args.cv: 
        train, val, test = make_full_df(args.checkpoint_folder, args.cv, args.fold_inx)
    else:
        train, test = make_full_df(args.checkpoint_folder,args.cv)

    if args.output_file == '':
        file_root = args.checkpoint_folder.split("/")[len(args.checkpoint_folder.split("/")) - 1]
    else:
        file_root = args.output_file
    
    train.to_csv(args.checkpoint_folder+ "/" + file_root + "_train.csv")
    test.to_csv(args.checkpoint_folder + "/" + file_root + "_test.csv")
    if args.cv:
        val.to_csv(args.checkpoint_folder + "/" + file_root + "_val.csv")        
        print(file_root + "{_train,_val,_test}.csv written to : " + args.checkpoint_folder)
    else:
        print(file_root + "{_train,_test}.csv written to : " + args.checkpoint_folder)


if __name__ == '__main__':
    main()
