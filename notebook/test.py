# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as stats
from itertools import combinations, permutations


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


def generate_candidate_pair(res, group_set, group_num, prefix, unit=50, tukey_test=False):

    def _group(i, prefix=prefix, unit=unit):
        return '{}-{}'.format((i - 1) * unit + 10 * (prefix - 1),   i * unit - 1 + 10 * (prefix - 1))

    if group_num == 2:
        sorted_uniformity = sorted(res.items(), key=lambda x: (x[1][0] + x[1][2], x[1][1]), reverse=True)
        for k, v in filter(lambda x: x[1][0] == 1 and x[1][2] == 1, sorted_uniformity):
            print('实验组：{}, 对照组：{}, p_val:{},{}'.format(_group(k[0]), _group(k[1]), v[1], v[3]))
    else:
        # 多组交叉验证
        final_res = dict()
        for multi_group in combinations(group_set.tolist(), group_num):
            # 1:生成2组查询检验结果for多重交叉验证【group_num * (group_num-1)/2】  testing 和单组验证【group_num-1】 testing
            cross_res, multi_res = [], dict()
            for binom_group in combinations(multi_group, 2):
                testing_res = res[binom_group]
                cross_res.append(testing_res)

                for group_id in binom_group:
                    val = multi_res[group_id] if group_id in multi_res else []
                    val.append(testing_res)
                    multi_res[group_id] = val

            sorted_multi_res = dict()
            for k, v in multi_res.items():
                sorted_multi_res[k] = np.average(np.array(v), axis=0).tolist()
            best_group = sorted(sorted_multi_res.items(), key=lambda x: (x[1][0] + x[1][2], x[1][1]), reverse=True)[0]
            final_res[multi_group] = [np.average(np.array(cross_res), axis=0).tolist(), [best_group[0], best_group[1]]]

        # 2:输出候选结果
        sorted_res = sorted(final_res.items(), key=lambda x: (x[1][0][0] + x[1][0][2], x[1][0][1]), reverse=True)
        cross_candidate_res = list(filter(lambda x: x[1][0][0] == 1 and x[1][0][2] == 1, sorted_res))
        if len(cross_candidate_res) == 0:
            # 重新排序
            sorted_res = sorted(final_res.items(), key=lambda x: (x[1][1][1][0] + x[1][1][1][2], x[1][1][1][1]),
                                reverse=True)
            print(">>> 组间多重交叉验证无候选，单组验证候选集合：")
            for k, v in filter(lambda x: x[1][1][1][0] == 1 and x[1][1][1][2] == 1, sorted_res):
                control_context_id = ["对照组{}：{}".format(index + 1, _group(i)) for index, i in enumerate(k[1:])]
                print('实验组：{}, {}'.format(_group(k[0]), ','.join(control_context_id)))
        else:
            print(">>> 组间多重交叉验证候选集合：")
            for k, v in cross_candidate_res:
                control_context_id = ["对照组{}：{}".format(index + 1, _group(i)) for index, i in enumerate(k[1:])]
                print('实验组：{}, {}'.format(_group(k[0]), ','.join(control_context_id)))

        # 3:输出tukey结果
        if tukey_test:
            print(">>> 多重比较矫正验证候选集合：")
            tukey_res = dict()
            for multi_group in combinations(group_set.tolist(), group_num):
                tukey_res[multi_group] = res[multi_group]
            sorted_uniformity = sorted(tukey_res.items(), key=lambda x: (x[1][0] + x[1][2], x[1][1]), reverse=True)
            for k, v in filter(lambda x: x[1][0] == 1 and x[1][2] == 1, sorted_uniformity):
                control_context_id = ["对照组{}：{}".format(index + 1, _group(i)) for index, i in enumerate(k[1:])]
                print('实验组：{}, {}'.format(_group(k[0]), ','.join(control_context_id)))
        return tukey_res
