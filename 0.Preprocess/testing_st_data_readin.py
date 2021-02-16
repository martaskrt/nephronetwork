import pandas as pd
import math
from skimage import exposure
import numpy as np
import argparse
import os
#import ast


def containsAll(str, set):
    """
    Check whether sequence str contains ALL of the items in set.
    Solution from: https://www.oreilly.com/library/view/python-cookbook/0596001673/ch03s07.html
    """
    return 0 not in [c in str for c in set]



def get_st_files(img_dir):
    out_dict = {"R": dict(), "L": dict()}
    for subdir, dirs, files in os.walk(img_dir):
        if any([containsAll(file, "-preprocced.png") for file in files]):
            us_num_files = [file for file in files if containsAll(file, "-preprocced.png")]

            us_nums = list(set([file.lower()[-19:-18] for file in us_num_files]))
            us_num_set = set(us_nums)

        for side in ["R", "L"]:
            out_dict[side] = dict.fromkeys(sorted(us_num_set), dict())

        print(out_dict)

        for file in us_num_files:
            num = file.lower()[-19:-18]
            print("US number: " + num)

            if file.lower()[-18:] == "r-preprocessed.png":
                if containsAll(file.lower(), "trv"):
                    out_dict["R"][num]["trv"] = os.path.join(subdir, file)
                elif containsAll(file.lower(), "sag"):
                    out_dict["R"][num]["sag"] = os.path.join(subdir, file)

            elif file.lower()[-18:] == "l-preprocessed.png":
                if containsAll(file.lower(), "trv"):
                    out_dict["L"][num]["trv"] = file
                elif containsAll(file.lower(), "sag"):
                    out_dict["L"][num]["sag"] = file

    return out_dict



def main(data_folder="C:/Users/lauren erdman/Desktop/kidney_img/HN/silent_trial/ImageOutput/HN Outputs", ## put these in the args
                 data_sheet="C:/Users/lauren erdman/Desktop/kidney_img/HN/silent_trial/SilentTrial_Datasheet.csv"):
    my_dat = pd.read_csv(data_sheet)

    img_dict = dict()
    for i in my_dat["ST.Study.ID"].unique():
        study_folder = data_folder + "/Study ID " + str(i) + "/"
        img_dict["StudyID"+str(i)] = get_st_files(img_dir=study_folder)


if __name__ == '__main__':
    main()
