# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime
from itertools import combinations, permutations

import scipy.stats as stats
from scipy.stats import t
from scipy.stats import chi2_contingency

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import warnings

warnings.filterwarnings('ignore')


def chi_square_test(table_array, alpha=0.05):
    """
    卡方检验
    """
    # stat卡方统计值，p：P_value，dof 自由度，expected理论频率分布(kf_arr)
    stat, p, dof, expected = chi2_contingency(table_array)
    if p < alpha:
        # print('reject H0:Dependent') # 拒绝原假设
        res = 0
    else:
        # print('fail to reject H0:Independent')  # 无法拒绝原假设
        res = 1
    return stat, p, res


def z_test(x1, x2, m1, m2, n1, n2, alpha=0.05):
    """
    z检验：用于比例类指标检验
    """
    x_all = (m1 + m2) / (n1 + n2)
    var = x_all * (1 - x_all)

    z_stat = (x1 - x2) / np.sqrt(var / n1 + var / n2)
    p = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    res = '显著' if p < alpha else '不显著，分流均匀'
    return z_stat, p, res


def t_test(x1, x2, var1, var2, n1, n2, alpha=0.05):
    """
    t检验
    """
    t_stat = (x1 - x2) / np.sqrt(var1 / n1 + var2 / n2)
    df = n1 + n2 - 2
    p = (1 - t.cdf(abs(t_stat), df)) * 2
    res = '显著' if p < alpha else '不显著，分流均匀'
    return t_stat, p, res


def generate_hypothesis_testing_result(data,
                                       test_column,
                                       n_column,
                                       m_column=None,
                                       var_column=None,
                                       alpha=0.05,
                                       print_log=False):
    """
    假设检验
    """
    p_res, success_num = [], 0

    for date in data.date.unique():
        test_df = data[(data.date == date)]

        if m_column is not None:
            # 比例指标的t检验
            test_df = test_df[['exp_group', test_column, m_column, n_column]]
            _, x1, m1, n1 = test_df[test_df['exp_group'] == '对照组'].values[0]
            _, x2, m2, n2 = test_df[test_df['exp_group'] == '实验组'].values[0]
            z_stat, p, test_res = z_test(x1, x2, m1, m2, n1, n2, alpha=alpha)

        elif var_column is not None:
            # 均值指标的t检验
            test_df = test_df[['exp_group', test_column, n_column, var_column]]
            _, x1, n1, var1 = test_df[test_df['exp_group'] == '对照组'].values[0]
            _, x2, n2, var2 = test_df[test_df['exp_group'] == '实验组'].values[0]
            t_stat, p, test_res = t_test(x1, x2, var1, var2, n1, n2, alpha=alpha)

        p_res.append(p)
        success_num += 0 if p < 0.05 else 1
        if print_log:
            print('日期:{date}, 指标:{col}, p:{p}, 检验结果:{stat}, diff:{diff:3%}'.format(date=date,
                                                                                   col=test_column,
                                                                                   p=p,
                                                                                   stat=test_res,
                                                                                   diff=x1 / x2 - 1))

    return p_res, success_num


if __name__ == '__main__':
    df = pd.read_excel('../data/hypothesis_data.xlsx')
    z_stat_res, p_res = [], []
    for line in df.itertuples():
        # print(line)
        _, n1, m1, n2, m2 = line
        z_stat, p, res = z_test(m1/n1, m2/n2, m1, m2, n1, n2)
        z_stat_res.append(z_stat)
        p_res.append(p)
    print(p_res)
        # print(p, res)
