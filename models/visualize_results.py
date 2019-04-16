
'''
Input is text file with the following format:
# TrainEpoch\tNUM\tACC\tNUM%\tLoss\tNUM\tAUC\tNUM\tAUPRC\tNUM\tTN\tNUM\tFP\tNUM\tFN\tNUM\tTP\tNUM
'''

import argparse


def load_data(inputfile):
    data = {}
    counter = 0
    with open(inputfile, 'r') as f:
        for line in f:
            content = line[:-1].split('\t')
            if counter == 0:
                for label in range(2, len(content), 2):
                    data[content[label]] = []
                counter += 1
            for item in range(3, len(content), 2):
                label = item-1
                data[content[label]].append(content[item])

    return data

def confusion_matrix(inputfile):
    data = load_data(inputfile)
    print(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', required=True, help='file to process. '
                                                       'format for each line in file: TrainEpoch\tNUM\tACC\tNUM%\tLoss'
                                                       '\tNUM\tAUC\tNUM\tAUPRC\tNUM\tTN\tNUM\tFP\tNUM\tFN\tNUM\tTP\tNUM')
if __name__ == "__main__":
    main()


