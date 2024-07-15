# -*- coding: utf-8 -*-

import yaml
import hashlib

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib

from scipy.stats import chi2_contingency


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
    decimal_digest = int(md.hexdigest()[:15], 16)
    bucket_id = decimal_digest % 1000
    return bucket_id


if __name__ == '__main__':
    # default args
    game_name = 'dnfm'
    layer_name = 'tx_dygame_dnfm_0614_layer1'
    # load data
    df = pd.read_csv('../data/tmp/dnfm_tx_dnfm_hashtest.csv')
    df.columns = ["test_id", "hash_key_id", "md5", "decimal_digest", "buckect_id"]
    # df['test_bucket_id'] = df['random_id'].apply(lambda x: generate_hash_group_id(x, game_name, layer_name))
    part1 = df.loc[:9999, :]
    part2 = df.loc[10000:,:]

    print(part1.shape, part2.shape)
    part1.to_csv('../data/tmp/dnfm_tx_dnfm_hashtest_001.csv', index=False)
    part2.to_csv('../data/tmp/dnfm_tx_dnfm_hashtest_002_res.csv', index=False)
