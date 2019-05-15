import numpy as np


def load_data(inputfile):
    file_names = inputfile.split(",")
    data = {}

    for file_ in file_names:
        with open(file_, 'r') as f:
            for fold in range(1, 6):
                data[fold] = {}
                data[fold]["train"] = {}
                data[fold]["val"] = {}
                data[fold]["test"] = {}
            counter = 0
            for line in f:
                content = line[:-1].split('\t')
                if len(content) < 2 or content[0] == 'ARGS':
                    continue
                if content[2] not in ["ValEpoch", "TrainEpoch", "TestEpoch"]:
                    continue
                fold = int(content[1])
                if counter == 0:
                    for label in range(4, len(content), 2):
                        for fold_ in range(1, 6):
                            data[fold_]["train"][content[label]] = []
                            data[fold_]["val"][content[label]] = []
                            data[fold_]["test"][content[label]] = []
                    counter += 1
                for item in range(5, len(content), 2):
                    label = item - 1
                    val = float(content[item])
                    if "Train" in content[2]:
                        data[fold]["train"][content[label]].append(val)
                    elif "Val" in content[2]:
                        data[fold]["val"][content[label]].append(val)
                    elif "Test" in content[2]:
                        data[fold]["test"][content[label]].append(val)
    return data


def get_metric_results(data, metric, early_stop_epoch):

    avg_of_val_epochs = np.array(data[1]['val'][metric])
    for fold in range(2, 6):
        avg_of_val_epochs = np.sum((avg_of_val_epochs, np.array(data[fold]['val'][metric])), axis=0)
    avg_of_val_epochs /= 5
    avg_of_val_epochs = np.array(avg_of_val_epochs)[:early_stop_epoch+1]

    max_of_avg_epochs = int(np.argmax(avg_of_val_epochs))

    print("best_val_epoch\t{}".format(max_of_avg_epochs))
    max_avg_epochs_results = {}
    max_avg_epochs_results['train'] = {}
    max_avg_epochs_results['val'] = {}
    max_avg_epochs_results['test'] = {}
    for metric in ["AUC", "AUPRC"]:
        max_avg_epochs_results['train'][metric] = 0
        max_avg_epochs_results['val'][metric] = 0
        max_avg_epochs_results['test'][metric] = 0
        for fold in range(1, 6):
            max_avg_epochs_results['train'][metric] += data[fold]['train'][metric][max_of_avg_epochs]
            max_avg_epochs_results['val'][metric] += data[fold]['val'][metric][max_of_avg_epochs]
            max_avg_epochs_results['test'][metric] += data[fold]['test'][metric][max_of_avg_epochs]

        max_avg_epochs_results['train'][metric] /= 5
        max_avg_epochs_results['val'][metric] /= 5
        max_avg_epochs_results['test'][metric] /= 5

    return max_avg_epochs_results


def compute_results(filename, data, early_stop_epoch):
    print("FILE NAME......................................" + str(filename))
    print("early_stop_epoch\t{}".format(early_stop_epoch))
    metrics = get_metric_results(data, 'AUC', early_stop_epoch)
    # auprc = get_metric_results(data, 'AUPRC', early_stop_epoch)
    print('{:.3f}/{:.3f}\t{:.3f}/{:.3f}\t{:.3f}/{:.3f}'.format(metrics['train']["AUC"], metrics['train']["AUPRC"],
                                                               metrics['val']["AUC"], metrics['val']["AUPRC"],
                                                               metrics['test']["AUC"], metrics['test']["AUPRC"]))
    print("**********************************************************************************************")

