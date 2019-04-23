'''
Input is text file with the following format:
# TrainEpoch\tNUM\tACC\tNUM%\tLoss\tNUM\tAUC\tNUM\tAUPRC\tNUM\tTN\tNUM\tFP\tNUM\tFN\tNUM\tTP\tNUM
'''

import argparse
# import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def load_data(rootdir):
    data = {}
    results_files = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.lower()[-4:] == ".txt":
                results_files.append(os.path.join(subdir, file))
    for file in files:
        with open(file, 'r') as f:
            data[file] = {}
            for fold in range(1, 6):
                data[file][fold] = {}
                data[file][fold]["train"] = {}
                data[file][fold]["val"] = {}
                data[file][fold]["test"] = {}
            counter = 0
            for line in f:
                content = line[:-1].split('\t')
                if content[2] not in ["ValEpoch", "TrainEpoch", "TestEpoch"]:
                    continue
                fold = int(content[1])
                if counter == 0:
                    for label in range(4, len(content), 2):
                        data[file][fold]["train"][content[label]] = []
                        data[file][fold]["val"][content[label]] = []
                        data[file][fold]["test"][content[label]] = []
                    counter += 1
                for item in range(5, len(content), 2):
                    label = item-1
                    val = float(content[item])
                    if "Train" in content[2]:
                        data[file][fold]["train"][content[label]].append(val)
                    elif "Val" in content[0]:
                        data[file][fold]["val"][content[label]].append(val)
                    elif "Test" in content[0]:
                        data[file][fold]["test"][content[label]].append(val)

        return data



def compute_results(args):
    data = load_data(args.dir)
    for file in data:
        max_indiv_epochs_results = {}
        max_indiv_epochs_results['train'] = 0
        max_indiv_epochs_results['val'] = 0
        max_indiv_epochs_results['test'] = 0
        for fold in range(1,6):
            data[file][fold]['val']["AUC"] = np.array(data[file][fold]['val']["AUC"])
            max_val_epoch = int(np.argmax(data[file][fold]['val']["AUC"]))
            max_indiv_epochs_results['train'] += data[file][fold]['train']["AUC"][max_val_epoch]
            max_indiv_epochs_results['val'] += data[file][fold]['val']["AUC"][max_val_epoch]
            max_indiv_epochs_results['test'] += data[file][fold]['test']["AUC"][max_val_epoch]

        max_indiv_epochs_results['train'] /= 5
        max_indiv_epochs_results['val'] /= 5
        max_indiv_epochs_results['test'] /= 5

        max_avg_epochs_results = {}
        max_avg_epochs_results['train'] = 0
        max_avg_epochs_results['val'] = 0
        max_avg_epochs_results['test'] = 0

        avg_of_val_epochs = np.array(data[file][1]['val']["AUC"])
        for fold in range(2, 6):
            avg_of_val_epochs = np.sum((avg_of_val_epochs,np.array(data[file][fold]['val']["AUC"])))
        avg_of_val_epochs /= 5
        max_avg_epochs_results = int(np.argmax(avg_of_val_epochs))

        for fold in range(1,6):
            max_avg_epochs_results['train'] += data[file][fold]['train']["AUC"][max_avg_epochs_results]
            max_avg_epochs_results['val'] += data[file][fold]['val']["AUC"][max_avg_epochs_results]
            max_avg_epochs_results['test'] += data[file][fold]['test']["AUC"][max_avg_epochs_results]

        max_avg_epochs_results['train'] /= 5
        max_avg_epochs_results['val'] /= 5
        max_avg_epochs_results['test'] /= 5

        print("FILE NAME......................................" + str(file))
        print("max_indiv_epochs_results:")
        print(max_indiv_epochs_results)
        print("max_avg_epochs_results:")
        print(max_avg_epochs_results)
        print("**********************************************************************************************")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', required=True, help='file to process. If multiple, comma separated '
                                                       'format for each line in file: TrainEpoch\tNUM\tACC\tNUM%\tLoss'
                                                       '\tNUM\tAUC\tNUM\tAUPRC\tNUM\tTN\tNUM\tFP\tNUM\tFN\tNUM\tTP\tNUM')
    args = parser.parse_args()
    compute_results(args)
if __name__ == "__main__":
    main()


