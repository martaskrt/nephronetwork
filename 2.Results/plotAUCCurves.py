import matplotlib.pyplot as plt
import sys
import json

def parseData(datapath, wantEpoch):
    f = open(datapath).read()
    parsed = f.split("{")
    dataStr = "{" + parsed[wantEpoch+1].split("}")[0] + "}"
    replace = {"'": '"',
        "0.": "0.0", "1.": "1.0",
        "\n": "",
        " ": "",
        "array(": "",
        ")": "",
        ",dtype=float32": "",
        ",...":""}
    for k,v in replace.items():
        dataStr = dataStr.replace(k,v)
    data = json.loads(dataStr)
    return data

def plotAUROC(fpr, tpr, thresholds):
    plt.plot(fpr, tpr)
    plt.show()

def plotAUPRC(precision, recall, thresholds):
    plt.plot(recall, precision)
    plt.show()

def main():
    wantEpoch = int(sys.argv[1])  # pass the epoch u want here
    path = sys.argv[2]  # pass in auc data file
    d = sys.argv[3] # pass in either test, or val; train does not currently work
    # d == "train", "test", or "val"

    data = parseData(path, wantEpoch)  # note data has thresholds if needed
    fpr, tpr = data[d+"_"+"fpr"], data[d+"_"+"tpr"]
    precision, recall = data[d+"_"+"precision"], data[d+"_"+"recall"]
    print("Displaying AUROC plot")
    plotAUROC(fpr, tpr, None)
    print("Displaying AUPRC plot")
    plotAUROC(precision, recall, None)

if __name__ == '__main__':
    main()