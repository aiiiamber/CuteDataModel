# -*-coding:utf-8 -*-
# @Author: xiaolizhang

import numpy as np
from sklearn.metrics import roc_auc_score


def cal_auc(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_true[y_true > 0] = 1
    y_true[y_true < 0] = 0

    score = roc_auc_score(y_true, y_pred)
    return score
