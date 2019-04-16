# add following line to top of script to import this module: load_dataset =
# process_results = importlib.machinery.SourceFileLoader('process_results','/path/to/models/dir/process_results.py').load_module()

from sklearn import metrics

def get_metrics(y_score, y_true, y_pred):

    fpr, tpr, thresholds = metrics.roc_curve(y_score=y_score, y_true=y_true)
    fnr = 1 - tpr
    tnr = 1 - fpr
    auc = metrics.auc(fpr, tpr)
    auprc = metrics.average_precision_score(y_score=y_score, y_true=y_true)

    tn, fp, fn, tp = metrics.confusion_matrix(y_pred=y_pred, y_true=y_true).ravel()

    results = {'fpr': fpr,
               'fp': fp,
               'tpr': tpr,
               'tp': tp,
               'fnr': fnr,
               'fn': fn,
               'tnr': tnr,
               'tn': tn,
               'auc': auc,
               'auprc': auprc}

    return results