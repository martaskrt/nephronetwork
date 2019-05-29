'''
Input is text file with the following format:
# TrainEpoch\tNUM\tACC\tNUM%\tLoss\tNUM\tAUC\tNUM\tAUPRC\tNUM\tTN\tNUM\tFP\tNUM\tFN\tNUM\tTP\tNUM
'''

import argparse
# import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def load_data(inputfile):
    file_names = inputfile.split(",")
    data = {}

    for file_ in file_names:
        # if "jigsaw" in file_:
        #     file = "jigsaw"
        # else:
        file = "vanilla"
        if file not in data:
            data[file] = {}
        with open(file_, 'r') as f:
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
                        for fold_ in range(1, 6):
                            data[file][fold_]["train"][content[label]] = []
                            data[file][fold_]["val"][content[label]] = []
                            data[file][fold_]["test"][content[label]] = []
                    counter += 1
                for item in range(5, len(content), 2):
                    label = item - 1
                    val = float(content[item])
                    if "Train" in content[2]:
                        data[file][fold]["train"][content[label]].append(val)
                    elif "Val" in content[2]:
                        data[file][fold]["val"][content[label]].append(val)
                    elif "Test" in content[2]:
                        data[file][fold]["test"][content[label]].append(val)
        # for fold in range(1, 6):
        #     for key in data[file][fold]["test"]:
        #         assert len(data[file][fold]["test"][key]) == 50
        #         assert len(data[file][fold]["train"][key]) == 50
        #         assert len(data[file][fold]["val"][key]) == 50
    return data

def confusion_matrix(args):
    data = load_data(args.fname)

    avg_loss = {'jigsaw': {},
                'vanilla': {} }

    # avg_loss['jigsaw']['train'] = np.array(data['jigsaw'][1]['train']["Loss"])
    avg_loss['vanilla']['train'] = np.array(data['vanilla'][1]['train']["Loss"])
    # avg_loss['jigsaw']['val'] = np.array(data['jigsaw'][1]['val']["Loss"])
    avg_loss['vanilla']['val'] = np.array(data['vanilla'][1]['val']["Loss"])

    for fold in range(2, 6):
        avg_loss['vanilla']['train'] = np.sum((avg_loss['vanilla']['train'],
                                               np.array(data['vanilla'][fold]['train']["Loss"])), axis=0)
        avg_loss['vanilla']['val'] = np.sum((avg_loss['vanilla']['val'],
                                               np.array(data['vanilla'][fold]['val']["Loss"])), axis=0)
        # avg_loss['jigsaw']['train'] = np.sum((avg_loss['jigsaw']['train'],
        #                                        np.array(data['jigsaw'][fold]['train']["Loss"])), axis=0)
    avg_loss['vanilla']['train'] /= 5
    # avg_loss['jigsaw']['train'] /= 5


    dataframe = {'vanilla-train': avg_loss['vanilla']['train'],
                 # 'jigsaw-train': avg_loss['jigsaw']['train'],
                 'vanilla-val': avg_loss['vanilla']['val']
                 # 'jigsaw-val': avg_loss['jigsaw']['val'],
                 }
    p_dataframe = pd.Series(dataframe)
    # ax = sns.scatterplot(x=np.arange(50), y=data["jigsaw"]["train"]["AUC"])
    plt.scatter(x=np.arange(50), y=p_dataframe['vanilla-train'], color='deepskyblue', label='vanilla')
    # plt.scatter(x=np.arange(50), y=p_dataframe['jigsaw-train'], color='slateblue', label='jigsaw')
    plt.scatter(x=np.arange(50), y=p_dataframe['vanilla-val'], color='gold', label='vanilla-val')
    # plt.scatter(x=np.arange(50), y=p_dataframe['jigsaw-val'], color='violet', label='jigsaw-val')
    # ax = sns.scatterplot(data=p_dataframe)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.legend()
    plt.show()




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', required=True, help='file to process. If multiple, comma separated '
                                                       'format for each line in file: TrainEpoch\tNUM\tACC\tNUM%\tLoss'
                                                       '\tNUM\tAUC\tNUM\tAUPRC\tNUM\tTN\tNUM\tFP\tNUM\tFN\tNUM\tTP\tNUM')
    args = parser.parse_args()
    confusion_matrix(args)
if __name__ == "__main__":
    main()