import pydicom
import argparse
import os
import pandas as pd

# loads all image paths to array
def load_file_names(dcms_dir):
    dcm_files = []
    for subdir, dirs, files in os.walk(dcms_dir):
        for file in files:
            if file.lower()[-4:] == ".dcm":
                dcm_files.append(os.path.join(subdir, file))
    return sorted(dcm_files)

def get_dcm_info(dcm_file):
    my_dcm = pydicom.dcmread(dcm_file)

    try:
        my_mrn = my_dcm.PatientID

    except AttributeError:
        try:
            my_mrn = my_dcm.ImageID

        except AttributeError:
            my_mrn = "NA"

    return my_mrn



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dcms_dir', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/all-cabs_pull20200322/us/1-1000/',
                        help="directory of US sequence dicoms")
    parser.add_argument('-csv_out_dir', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/all-cabs_pull20200322/',
                        help="directory of US sequence dicoms")
    # parser.add_argument('-dcms_dir', default='C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Front-end-test-files/test_us_dmsa/',
    #                     help="directory of US sequence dicoms")
    # parser.add_argument('-csv_out_dir', default='C:/Users/larun/Desktop/Data Science Core/Projects/Urology/Front-end-test-files/test_us_dmsa/',
    #                     help="directory of US sequence dicoms")
    parser.add_argument('-csv_filename', default='func_us_pull_20200401_dicom_mrns.csv',
                        help="MRN csv filename")
    parser.add_argument('-csv_file_extension', default='NA',
                        help="MRN csv filename")

    opt = parser.parse_args() ## comment for debug

    print("dcms_dir: " + opt.dcms_dir)

    my_dcm_files = load_file_names(dcms_dir=opt.dcms_dir)

    print(my_dcm_files)

    my_mrns = [get_dcm_info(ind_file) for ind_file in my_dcm_files]

    unique_mrns = list(set(my_mrns))

    mrn_df = pd.DataFrame({"MRN": unique_mrns})

    print("Writing csv file to: " + opt.csv_out_dir + "/" + opt.csv_file_extension + opt.csv_filename)
    mrn_df.to_csv(opt.csv_out_dir + "/" + opt.csv_file_extension + opt.csv_filename)


if __name__ == "__main__":
    main()

