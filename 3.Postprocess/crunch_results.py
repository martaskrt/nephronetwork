
import numpy as np
import process_model_output_file
import argparse
import os

def get_fold_avg(data_dict):
    epoch_avg_per_fold = {'train': [],
                          'val': [],
                          'test': []}
    for key in data_dict[1]:
        epoch_avg_per_fold[key] = np.array(data_dict[1][key]["Loss"])
        for fold in range(2, 6):
            epoch_avg_per_fold[key] = np.sum((epoch_avg_per_fold[key], np.array(data_dict[fold][key]["Loss"])), axis=0)
        epoch_avg_per_fold[key] /= 5
    return epoch_avg_per_fold

def get_best_epoch(valloss, run=5):
    min_loss = 0
    min_loss_epoch = 0
    found_it = True
    for i in range(len(valloss)-run+1):
        found_it = True
        curr_min = valloss[i]
        for j in range(i+1, i+run):
            if valloss[j]-0.000 <= curr_min:
                found_it = False
        if found_it:
            min_loss = curr_min
            min_loss_epoch = i
            found_it = True
            break
    if not found_it:
        counter = run-1
        while counter > 0:
            i = len(valloss)-counter
            found_it = True
            curr_min = valloss[i]
            for j in range(i + 1, i + counter):
                if valloss[j] <= curr_min:
                    found_it = False
            if found_it:
                min_loss = curr_min
                min_loss_epoch = i
                found_it = True
                break
            counter -= 1
        if not found_it:
            min_loss_epoch = len(valloss)-1
    best_valloss = min_loss_epoch
    return best_valloss


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--fname', required=True, help='file to process. If multiple, comma separated '
    #                                                    'format for each line in file: TrainEpoch\tNUM\tACC\tNUM%\tLoss'
    #                                                    '\tNUM\tAUC\tNUM\tAUPRC\tNUM\tTN\tNUM\tFP\tNUM\tFN\tNUM\tTP\tNUM')
    parser.add_argument('--dir', help='path/to/dir/with/results')
    args = parser.parse_args()

    results_files = []
    for subdir, dirs, files in os.walk(args.dir):
        for file in files:
            if file.lower()[-4:] == ".txt":
                results_files.append(os.path.join(subdir, file))

    for file in results_files:
        # try:
        print(file)
        data = process_model_output_file.load_data(file)
        metric = 'Loss'

        #for fold in range(1,6):
         #   print(len(np.array(data[fold]['val'][metric])))
        valloss = np.array(data[1]['val'][metric])
        for fold in range(2, 6):
            valloss = np.sum((valloss, np.array(data[fold]['val'][metric])), axis=0)
        #print(valloss)
        early_stop_epoch = get_best_epoch(valloss, run=10)
        # early_stop_epoch = 17
        process_model_output_file.compute_results(file, data, early_stop_epoch)
        # except:
        #     data = process_model_output_file.load_data(file, stop=9)
        #
        #     metric='Loss'
        #     valloss = np.array(data[1]['val'][metric])
        #
        #     for fold in range(2, 6):
        #         valloss = np.sum((valloss, np.array(data[fold]['val'][metric])), axis=0)
        #
        #     early_stop_epoch = get_best_epoch(valloss, run=10)
        #     # early_stop_epoch=17
        #     process_model_output_file.compute_results(file, data, early_stop_epoch)


    # results_files = []
    # for subdir, dirs, files in os.walk(rootdir):
    #     for file in files:
    #         if file.lower()[-4:] == ".txt":
    #             results_files.append(os.path.join(subdir, file))


if __name__ == "__main__":
    main()