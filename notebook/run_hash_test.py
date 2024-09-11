# -*- coding: utf-8 -*-

import yaml
import hashlib

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib

import collections
from statsmodels.stats.libqsturng import qsturng

from scipy.stats import chi2_contingency


def tukeyHSD(info, alpha=0.05):
    """
    tukey's method with summary statistics

    params:
    - info: list, each element to be the mean, variance and sample size
    of each version. e.g. [(mean, variance, sample_size),...]
    - alpha: float, FWER control, 0.05 by default

    returns:
    - HSD: dictionary, contains
        - adjusted confidence interval
        - whether to reject
      for each pair-wise comparison
    """

    # number of variants
    k = len(info)
    # Total samples
    n = sum([_[2] for _ in info])
    # average sample per variant
    r = int(n / k)
    # critical value: studentized range
    Q = qsturng(1 - alpha, k, n - k)
    D = Q / np.sqrt(r)
    # sum of squares
    SSE = sum((_[2] - 1) * _[1] for _ in info)
    MSE = float(SSE) / (n - k)
    # adjusted margin
    margin = D * np.sqrt(MSE)
    # tukey's honest significance test
    HSD = collections.defaultdict(dict)
    # pair-wise comparison
    for base_idx in range(len(info)):
        for trt_idx in range(base_idx + 1, len(info)):
            m_diff = -(info[base_idx][0] - info[trt_idx][0])
            HSD[str(base_idx) + '-' + str(trt_idx)]['CI'] = (m_diff - margin, m_diff + margin)
            if m_diff - margin > 0 or m_diff + margin < 0:
                HSD[str(base_idx) + '-' + str(trt_idx)]['reject'] = True
            else:
                HSD[str(base_idx) + '-' + str(trt_idx)]['reject'] = False
    return HSD


def chi_square_test(df, indicator):
    """
    Chi-Square
    :param df: test data，should include：indicator column and indicator expected column
    :param indicator: test indicator
    :return:
    """
    # build table
    table = df[[indicator, indicator + '_expected']].values
    # stat卡方统计值，p：P_value，dof 自由度，expected理论频率分布(kf_arr)
    stat, p, dof, expected = chi2_contingency(table)
    # 选取95%置信度
    prob = 0.95
    alpha = 1 - prob
    # print('significance=%.3f,p=%.3f'%(alpha,p))
    if p < alpha:
        # 拒绝原假设
        # print('reject H0:Dependent')
        return 0
    else:
        # 无法拒绝原假设,则实际分配到桶中的样本数和预期一致，通过卡方检验
        # print('fail to reject H0:Independent')
        return 1


def generate_hash_group_id(id, game_name, layer_name):
    hash_key_id = "{}:{}:{}".format(id, game_name, layer_name)
    md = hashlib.md5()
    md.update(hash_key_id.encode('utf8'))
    md5_val = md.hexdigest()
    decimal_digest = int(md5_val[:15], 16)
    bucket_id = decimal_digest % 1000
    res = [id, hash_key_id, md5_val, decimal_digest, bucket_id]
    return res


if __name__ == '__main__':
    # default args
    game_name = 'wangzhe'
    layer_name = 'tx_dygame_wangzhe_0715_layer1'
    mode = 'test'  # generate, test

    if mode == 'generate':
        # Step1：取原始随机生成的ID
        df = pd.read_csv('../data/tmp/dnfm_tx_dnfm_hashtest.csv')
        df.columns = ["test_id", "hash_key_id", "md5", "decimal_digest", "buckect_id"]
        samples = df['test_id'].values
        test_data = []
        for id in samples:
            test_data.append(generate_hash_group_id(id, game_name=game_name, layer_name=layer_name))
        res = pd.DataFrame(test_data, columns=["test_id", "hash_key_id", "md5", "decimal_digest", "buckect_id"])

        # Step2：切分数据集
        part1 = res.loc[:9999, :]
        part2 = res.loc[10000:, 'test_id']
        part3 = res.loc[10000:, :]
        print(part1.shape, part2.shape, part3)

        # Step3：存储三份数据
        part1.to_csv('../data/tmp/{}_dygame_tx_hashtest_001.csv'.format(game_name), index=False)
        part2.to_csv('../data/tmp/{}_dyagme_tx_hashtest_002.csv'.format(game_name), index=False)
        part3.to_csv('../data/tmp/{}_dyagme_tx_hashtest_002_res.csv'.format(game_name), index=False)
    elif mode == 'test':
        return_res = pd.read_csv('../data/tmp/王者侧哈希结果.csv')
        local_res = pd.read_csv('../data/tmp/{}_dyagme_tx_hashtest_002_res.csv'.format(game_name))
        print('Different rate: {:2%}'.format((local_res['buckect_id'] - return_res['buckect_id']).sum()))

