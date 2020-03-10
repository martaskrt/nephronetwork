import os
import pandas as pd
import argparse

# loads all image paths to array
def load_file_names(my_dir,my_file_string = ""):
    cab_files = []
    for subdir, dirs, files in os.walk(my_dir):
        for file in files:
            if file.lower()[-len(my_file_string):] == my_file_string:
                cab_files.append(os.path.join(subdir, file))
    return sorted(cab_files)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-rootdir', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/data/load_training_test_sets/',
                        help="directory of US sequence dicoms")
    parser.add_argument('-cabs_rootdir', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/all-cabs/',
                        help="directory of US sequence dicoms")
    parser.add_argument('-csv_out_dir', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/all_label_csv/',
                        help="directory of US sequence dicoms")
    parser.add_argument('-id_linking_filename', default='full_linking_log_20200211.csv',
                        help="directory of US sequence dicoms")
    parser.add_argument('-cab_out_list_filename', default='/hpf/largeprojects/agoldenb/lauren/Hydronephrosis/data/load_training_test_sets/cab_list.csv',
                        help="directory of US sequence dicoms")
    parser.add_argument("--contrast", default=1, type=int, help="Image contrast to train on")

    opt = parser.parse_args() ## comment for debug

    print("All cabs directory: " + opt.cabs_rootdir)

    my_cab_files_in = load_file_names(my_dir=opt.cabs_rootdir, my_file_string=".cab")
    my_cab_files = [file.split("/")[-1] for file in my_cab_files_in]
    my_csv_files_in = load_file_names(my_dir=opt.csv_out_dir, my_file_string=".csv")
    my_csv_files = [file.split("/")[-1] for file in my_csv_files_in]
    my_linking_log = pd.read_csv(opt.rootdir + '/' + opt.id_linking_filename)

    csv_study_ids = [file_name[0:4] for file_name in my_csv_files]
    csv_string_ends = [file_name[4:] for file_name in my_csv_files]
    csv_string_id_ints = [my_id for my_id in csv_study_ids if len(my_id) == 4]
    # print(my_linking_log["deid"].values)
    csv_study_mrns = [my_linking_log.loc[my_linking_log["deid"] == int(my_id)].mrn.values for my_id in csv_string_id_ints if int(my_id) in my_linking_log["deid"].values]

    if len(csv_study_mrns) == len(csv_string_ends):
        # processed_cabs = ["D" + str(csv_study_mrns[i]) + str(csv_string_ends[i][0:-4]) + ".cab" for i in range(len(csv_study_mrns))]
        processed_cabs = ["D" + str(csv_study_mrns[i][0]) + str(csv_string_ends[i][0:-4]) + ".cab" for i in range(len(csv_study_mrns))]
        unprocessed_cabs = list(set(my_cab_files) - set(processed_cabs))
        # print(processed_cabs)
        # print(my_cab_files)
        # full_unprocessed_cabs = [opt.cabs_rootdir + file for file in unprocessed_cabs]
        # full_unprocessed_cabs = [opt.cabs_rootdir + file for file in unprocessed_cabs]
        print("Unprocessed cabs: ")
        print(unprocessed_cabs)
        cab_df = pd.DataFrame({"file_name": unprocessed_cabs})
        cab_df.to_csv(opt.cab_out_list_filename, header=False, index=False)
        print(opt.cab_out_list_filename + " created.")

    else:
        print("Error creating processed cabs list")

if __name__ == "__main__":
    main()




