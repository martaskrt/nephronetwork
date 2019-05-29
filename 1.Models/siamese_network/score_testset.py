'''
Input is text file with the following format:
# TrainEpoch\tNUM\tACC\tNUM%\tLoss\tNUM\tAUC\tNUM\tAUPRC\tNUM\tTN\tNUM\tFP\tNUM\tFN\tNUM\tTP\tNUM
'''

import argparse
# import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
import os


early_stopping = {
    "baseline_models_CV_results/fratnet_20190420_vanilla_CV_lr0.001_e50_bs256_SGD_v2.txt": 30,
    "baseline_models_CV_results/fratnet_20190420_vanilla_CV_lr0.001_e50_bs256_c1_SGD_v2.txt": 30,
    "baseline_models_CV_results/fratnet_20190420_vanilla_CV_lr0.001_e50_bs256_c2_SGD_v2.txt": 30,

    "baseline_models_CV_results/siamnet_20190420_vanilla_CV_lr0.001_e50_bs256_SGD_v2.txt": 33,
    "baseline_models_CV_results/siamnet_20190420_vanilla_CV_lr0.001_e50_bs256_c1_SGD_v2.txt": 33,
    "baseline_models_CV_results/siamnet_20190420_vanilla_CV_lr0.001_e50_bs256_c2_SGD_v2.txt": 33,

    "baseline_models_CV_results/siamnet_20190420_vanilla_CV_lr0.005_e50_bs256_c1_SGD_v2.txt": 16,
    "baseline_models_CV_results/siamnet_20190420_vanilla_CV_lr0.003_e50_bs256_c1_SGD_v2.txt": 15,

    "baseline_models_CV_results/siamnet_20190420_vanilla_CV_lr0.0001_e50_bs256_SGD_v2.txt": 50,
    "baseline_models_CV_results/siamnet_20190420_vanilla_CV_lr0.0001_e50_bs256_c1_SGD_v2.txt": 21,
    "baseline_models_CV_results/siamnet_20190420_vanilla_CV_lr0.0001_e50_bs256_c2_SGD_v2.txt": 22,

    "baseline_models_CV_results/siamnet_20190420_vanilla_CV_lr0.001_e50_bs256_ADAM_v2.txt": 5,
    "baseline_models_CV_results/siamnet_20190420_vanilla_CV_lr0.001_e50_bs256_c1_ADAM_v2.txt": 5,
    "baseline_models_CV_results/siamnet_20190420_vanilla_CV_lr0.001_e50_bs256_c2_ADAM_v2.txt": 5,

    "baseline_models_CV_results/siamnet_20190420_vanilla_CV_lr0.001_e50_bs256_SGD_m0.99_v2.txt": 17,
    "baseline_models_CV_results/siamnet_20190420_vanilla_CV_lr0.001_e50_bs256_c1_SGD_m0.99_v2.txt": 22,
    "baseline_models_CV_results/siamnet_20190420_vanilla_CV_lr0.001_e50_bs256_c2_SGD_m0.99_v2.txt": 15,

    # "baseline_models_CV_results/siamnet_20190420_jigsawe30lr0.005s64c1bn_CV_lr0.001_e50_bs256_c1_SGD_v2.txt": 35,
    # "baseline_models_CV_results/siamnet_20190420_jigsawe30lr0.01s64c1_CV_lr0.001_e50_bs256_c1_SGD_v2.txt": 23,
    # "baseline_models_CV_results/siamnet_20190420_jigsawe70lr0.01s64c1_CV_lr0.001_e50_bs256_c1_SGD_v2.txt": 33

    "baseline_models_CV_results/siamnet_20190420_jigsawe30lr0.005s64c1bn_CV_lr0.001_e50_bs256_c1_SGD_v2.txt": 31, #35,
    "baseline_models_CV_results/siamnet_20190420_jigsawe30lr0.005s64c1bn_CV_lr0.003_e50_bs256_c1_SGD_v2.txt": 31,
    "baseline_models_CV_results/siamnet_20190420_jigsawe30lr0.005s64c1bn_CV_lr0.005_e50_bs256_c1_SGD_v2.txt": 31,
    "baseline_models_CV_results/siamnet_20190420_jigsawe30lr0.005s64c1bn_CV_lr0.01_e50_bs256_c1_SGD_v2.txt": 11,
    "baseline_models_CV_results/siamnet_20190420_jigsawe30lr0.01s64c1_CV_lr0.001_e50_bs256_c1_SGD_v2.txt": 33, #29,21,
    "baseline_models_CV_results/siamnet_20190420_jigsawe70lr0.005s64c1bn_CV_lr0.003_e50_bs256_c1_SGD_v2.txt": 31,
    "baseline_models_CV_results/siamnet_20190420_jigsawe70lr0.01s64c1_CV_lr0.001_e50_bs256_c1_SGD_v2.txt": 39
}


def load_data(rootdir):
    data = {}
    results_files = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.lower()[-4:] == ".txt":
                results_files.append(os.path.join(subdir, file))
    for file in results_files:
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
                if len(content) < 2:
                    continue
                if content[2] not in ["ValEpoch", "TrainEpoch", "TestEpoch"]:
                    continue
                fold = int(content[1])
                if counter == 0:
                    for label in range(4, len(content), 2):
                        for fold_ in range(1,6):
                            data[file][fold_]["train"][content[label]] = []
                            data[file][fold_]["val"][content[label]] = []
                            data[file][fold_]["test"][content[label]] = []
                    counter += 1
                for item in range(5, len(content), 2):
                    label = item-1
                    val = float(content[item])

                    if "Train" in content[2] and len(data[file][fold]["train"][content[label]]) < early_stopping[file]:
                        data[file][fold]["train"][content[label]].append(val)
                    elif "Val" in content[2] and len(data[file][fold]["val"][content[label]]) < early_stopping[file]:
                        data[file][fold]["val"][content[label]].append(val)
                    elif "Test" in content[2] and len(data[file][fold]["test"][content[label]]) < early_stopping[file]:
                        data[file][fold]["test"][content[label]].append(val)
        # for fold in range(1, 6):
        #     for key in data[file][fold]["test"]:
        #         assert len(data[file][fold]["test"][key]) == 50
        #         assert len(data[file][fold]["train"][key]) == 50
        #         assert len(data[file][fold]["val"][key]) == 50
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
            avg_of_val_epochs = np.sum((avg_of_val_epochs, np.array(data[file][fold]['val']["AUC"])),axis=0)
        avg_of_val_epochs /= 5
        max_of_avg_epochs = int(np.argmax(avg_of_val_epochs))

        for fold in range(1,6):
            max_avg_epochs_results['train'] += data[file][fold]['train']["AUC"][max_of_avg_epochs]
            max_avg_epochs_results['val'] += data[file][fold]['val']["AUC"][max_of_avg_epochs]
            max_avg_epochs_results['test'] += data[file][fold]['test']["AUC"][max_of_avg_epochs]

        max_avg_epochs_results['train'] /= 5
        max_avg_epochs_results['val'] /= 5
        max_avg_epochs_results['test'] /= 5

        print("FILE NAME......................................" + str(file))
        print("max_indiv_epochs_results:")
        print('{:.3f}/{:.3f}/{:.3f}'.format(max_indiv_epochs_results['train'], max_indiv_epochs_results['val'],
                                            max_indiv_epochs_results['test']))
        print("max_avg_epochs_results:")
        print('{:.3f}/{:.3f}/{:.3f}'.format(max_avg_epochs_results['train'], max_avg_epochs_results['val'],
                                            max_avg_epochs_results['test']))
        print("**********************************************************************************************")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, help='file to process. If multiple, comma separated '
                                                       'format for each line in file: TrainEpoch\tNUM\tACC\tNUM%\tLoss'
                                                       '\tNUM\tAUC\tNUM\tAUPRC\tNUM\tTN\tNUM\tFP\tNUM\tFN\tNUM\tTP\tNUM')
    args = parser.parse_args()
    compute_results(args)
if __name__ == "__main__":
    main()


