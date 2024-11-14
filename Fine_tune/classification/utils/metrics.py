import numpy as np
from sklearn import metrics
import json
import os

def Specificity(output, target):
    con_mat = confusion_matrix(output, target)
    spe = []
    n = con_mat.shape[0]
    for i in range(n):
        number = np.sum(con_mat[:,:])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        fp = np.sum(con_mat[:,i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spe.append(spe1)
    return np.array(spe)
    
def AUC(output, target):
    
    assert output.shape[0] == target.shape[0]  # 
    class_auc = []
    for i in range(output.shape[1]):
        y_true = (target == i)
        y_pred = output[:, i]
        if np.max(y_true) > 0:
            class_auc.append(metrics.roc_auc_score(y_true, y_pred))
        else:
            class_auc.append(0)
    return np.array(class_auc)

def ACC(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.accuracy_score(y_true, y_pred)

def Cohen_Kappa(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.cohen_kappa_score(y_true, y_pred)

def F1_score(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    label = [i for i in range(output.shape[1])]
    return metrics.f1_score(y_true, y_pred, labels=label,average=None)

def Recall(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    label = [i for i in range(output.shape[1])]
    return metrics.recall_score(y_true, y_pred, labels=label,average=None)

def Precision(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    label = [i for i in range(output.shape[1])]
    return metrics.precision_score(y_true, y_pred, labels=label,average=None)

def cls_report(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.classification_report(y_true, y_pred, digits=4)


def confusion_matrix(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    label = [i for i in range(output.shape[1])]
    return metrics.confusion_matrix(y_true, y_pred,labels=label)

def write_score2json(score_info, val_anno_file, results_dir):
    score_info = score_info.astype(np.float)
    score_list = []
    anno_info = np.loadtxt(val_anno_file, dtype=np.str_)
    for idx, item in enumerate(anno_info):
        id = item[0].rsplit('/', 1)[-1]
        label = int(item[1])
        score = list(score_info[idx])
        pred = score.index(max(score))
        pred_info = {
            'image_id': id,
            'label': label,
            'prediction': pred,
            # 'benign/maglinant': int(pred in [1,3,6]),
            'score': score,
        }
        score_list.append(pred_info)
    json_data = json.dumps(score_list, indent=4)
    # file = open(os.path.join(results_dir, 'score_preds_unifS-B_mixcutc.json'), 'w')
    # file.write(json_data)
    # file.close()
