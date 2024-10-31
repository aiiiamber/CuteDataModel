# -*- coding: utf-8 -*-
import sys
import warnings

import math
import numpy as np

from scipy.stats import kendalltau, stats, spearmanr

from sklearn.metrics import roc_auc_score, confusion_matrix


def cal_auc(y_true, y_pred):
    assert (len(y_true) == len(y_pred))
    if len(np.unique(y_true)) < 2:
        warnings.warn('No positive sample or no negative sample in y_true!')

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_true[y_true > 0] = 1
    y_true[y_true < 0] = 0

    score = roc_auc_score(y_true, y_pred)

    return score


def cal_confusion_matrix(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall, TN, FP, FN, TP


def cal_gini(y_true, y_pred):
    assert (len(y_true) == len(y_pred))
    if len(np.unique(y_true)) < 2:
        warnings.warn('No positive sample or no negative sample in y_true!')

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # sort scores and corresponding truth values, desc
    desc_score_indices = np.argsort(y_pred)[::-1]
    y_true = y_true[desc_score_indices]
    y_pred = y_pred[desc_score_indices]

    # get x array and y array
    cumulative_actual = np.cumsum(y_true) / sum(y_true)  # normalized
    cumulative_index = np.arange(1, len(cumulative_actual) + 1) / len(y_pred)
    y_array = [0] + list(cumulative_actual)
    x_array = [0] + list(cumulative_index)

    # calculate gini coefficient
    area_b = abs(y_array[-1] * x_array[-1] - np.trapz(y_array, x_array))
    area_a = abs(y_array[-1] * x_array[-1] / 2 - area_b)
    gini_index = area_a / (area_a + area_b)

    return gini_index, (x_array, y_array)


def cal_norm_gini(y_true, y_pred):
    abs_gini, _ = cal_gini(y_true, y_pred)
    base_gini, _ = cal_gini(y_true, y_true)

    return abs_gini / base_gini


def cal_kendall(y_true, y_pred):
    assert (len(y_true) == len(y_pred))
    if len(np.unique(y_true)) < 2:
        warnings.warn('No positive sample or no negative sample in y_true!')

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tau, p_val = kendalltau(y_true, y_pred)

    return tau, p_val


def cal_spearman(y_true, y_pred):
    assert (len(y_true) == len(y_pred))
    if len(np.unique(y_true)) < 2:
        warnings.warn('No positive sample or no negative sample in y_true!')

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rho, p_val = spearmanr(y_true, y_pred)

    return rho, p_val


def cal_mae(label, score, percentile=None, breakers=None):
    label, score = np.array(label), np.array(score)

    if percentile is None and breakers is None:
        return np.mean(np.abs(label - score))

    if breakers is None:
        breakers = [0] + list(np.percentile(score, percentile)) + [sys.maxsize]

    def _mae(truth, pred):
        truth_mean = np.mean(truth)
        pred_mean = np.mean(pred)
        return np.abs(truth_mean - pred_mean), None

    return _cal_index_for_each_group(label, score, breakers, _mae)


def cal_mape(label, score, percentile=None, breakers=None):
    label, score = np.array(label), np.array(score)

    q = [10 * i for i in range(1, 10)]
    if percentile is not None:
        q = percentile

    if breakers is None:
        breakers = [0] + list(np.percentile(score, q)) + [sys.maxsize]

    def _mape(truth, pred):
        err = None

        truth_sum = sum(truth)
        pred_sum = sum(pred)

        if abs(truth_sum) < 1e-6:
            err = 'truth is too small %s' % err

        return np.abs(pred_sum - truth_sum) / abs(truth_sum), err

    return _cal_index_for_each_group(label, score, breakers, _mape)


def cal_recall(y_true, y_pred):
    assert (len(y_true) == len(y_pred))
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    cont = np.stack((y_true, y_pred), axis=-1)
    cont = cont[(-cont[:, 1]).argsort()]  # order desc by y_pred
    pos_index = np.where(cont[:, 0] > 0)[0]  # index of positive sample
    if len(pos_index) < 10:
        return "{} samples is not enough for evaluate recall".format(len(pos_index))
    pos_percentiles = np.arange(0.1, 1.1, 0.1)
    data_percentiles = []
    for pp in pos_percentiles:
        pp_index = int(len(pos_index) * pp)
        data_index = pos_index[pp_index - 1]
        percentile = (data_index + 1) / len(cont)
        data_percentiles.append(percentile)
    comp_percentiles = np.stack((pos_percentiles, data_percentiles), axis=-1).tolist()
    comp_str = "\n".join(["{:.2f},{:.5f}".format(c[0], c[1]) for c in comp_percentiles])
    res = "pos_cnt={}, total_cnt={}, percentiles:\n{}".format(len(pos_index), len(y_true), comp_str)
    return res


def _cal_index_for_each_group(label, score, breakers, fn):
    index_list = []

    for i in range(len(breakers) - 1):

        index = np.logical_and(score > breakers[i], score <= breakers[i + 1])
        truth = label[index]
        pred = score[index]

        index, err = fn(truth, pred)
        if err is not None:
            print('In segment %s: %s' % (i, err))
        else:
            index_list.append(index)

    return np.mean(index_list)


def ks_test(data1, data2, feature_names, alpha=0.05):
    pass_num = 0
    for feat in feature_names:
        sample1, sample2 = data1[feat], data2[feat]
        statistic, pvalue = stats.ks_2samp(sample1, sample2)
        if pvalue > alpha:
            pass_num += 0
            # print("Can not reject H0, two samples of {} may come from the same distribution.".format(feat))
        else:
            print("Reject H0(p_val:{}), two samples，two samples of {} may come from different distributions.".format(
                pvalue, feat))
    # balanced result
    if pass_num == len(feature_names):
        print("Pass Kolmogorow-Smirnov test, two samples come from the same distribution")
    else:
        ValueError('Two samples may come from different distribution')


def cal_cohenD(data1, data2, metricName):
    """
    Standardized Mean differences
    """
    treated_metric = data1[metricName]
    untreated_metric = data2[metricName]
    # calculate denominator
    denominator = math.sqrt(
        ((treated_metric.count() - 1) * treated_metric.std() ** 2 + (untreated_metric.count() - 1)
         * untreated_metric.std() ** 2) / (treated_metric.count() + untreated_metric.count() - 2))
    # if denominator is 0 divide by very small number
    if denominator == 0:
        d = abs((treated_metric.mean() - untreated_metric.mean()) / 0.000001)
    else:
        d = abs((treated_metric.mean() - untreated_metric.mean()) / denominator)
    return d
