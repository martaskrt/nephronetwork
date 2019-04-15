# add following line to top of script to import this module: load_dataset =
# process_results = importlib.machinery.SourceFileLoader('process_results','/path/to/models/dir/process_results.py').load_module()

from sklearn import metrics

def get_metrics(y_pred, y_true):

    fpr, tpr, thresholds = metrics.roc_curve(y_score=y_pred, y_true=y_true)
    fnr = 1 - tpr
    tnr = 1 - fpr
    auc = metrics.auc(fpr, tpr)
    auprc = metrics.average_precision_score(y_score=y_pred, y_true=y_true)

    results = {'fpr': fpr,
               'tpr': tpr,
               'fnr': fnr,
               'tnr': tnr,
               'auc': auc,
               'auprc': auprc}

    return results