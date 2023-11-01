import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


# # traditional non-effort-aware performance measures
def get_measure(y_true, y_pred):
    # y_pred[y_pred >= 0.5] = 1
    # y_pred[y_pred < 0.5] = 0

    AUC = roc_auc_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    temp = np.array([tp + fn, tp + fp, fn + tn, fp + tn]).prod()
    if temp != 0:
        MCC = (tp * tn - fn * fp) / np.sqrt(temp)
    else:
        MCC = 0

    return [AUC, MCC]
