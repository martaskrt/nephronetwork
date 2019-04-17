
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

    for file in file_names:
        if "jigsaw" in file:
            model = "jigsaw"
        else:
            model = "non-jigsaw"
        if model not in data:
            data[model] = {}

        with open(file, 'r') as f:
            data[model]["train"] = {}
            data[model]["val"] = {}
            counter = 0
            for line in f:
                content = line[:-1].split('\t')
                if content[0] not in ["ValEpoch", "TrainEpoch"]:
                    continue
                if counter == 0:
                    for label in range(2, len(content), 2):
                        data[model]["train"][content[label]] = []
                        data[model]["val"][content[label]] = []
                    counter += 1
                for item in range(3, len(content), 2):
                    label = item-1
                    if "%" not in content[item]:
                        val = float(content[item])
                    else:
                        val = content[item]
                    if "Train" in content[0]:
                        data[model]["train"][content[label]].append(val)
                    elif "Val" in content[0]:
                        data[model]["val"][content[label]].append(val)

    return data

def confusion_matrix(args):
    data = load_data(args.fname)
    dataframe = {'jigsaw-train': data["jigsaw"]["train"]["AUPRC"],
                 'jigsaw-val': data["jigsaw"]["val"]["AUPRC"],
                 'control-train': data["non-jigsaw"]["train"]["AUPRC"],
                 'control-val': data["non-jigsaw"]["val"]["AUPRC"]}
    p_dataframe = pd.Series(dataframe)
    # ax = sns.scatterplot(x=np.arange(50), y=data["jigsaw"]["train"]["AUC"])
    plt.scatter(x=np.arange(50), y=p_dataframe['jigsaw-train'], color='deepskyblue', label='jigsaw-train')
    plt.scatter(x=np.arange(50), y=p_dataframe['jigsaw-val'], color='gold', label='jigsaw-val')
    plt.scatter(x=np.arange(50), y=p_dataframe['control-train'], color='slateblue', label='control-train')
    plt.scatter(x=np.arange(50), y=p_dataframe['control-val'], color='violet', label='control-val')
    # ax = sns.scatterplot(data=p_dataframe)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('AUPRC', fontsize=18)
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


