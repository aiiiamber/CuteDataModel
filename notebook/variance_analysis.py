# -*- coding: utf-8 -*-
import yaml
import math

import numpy as np
import pandas as pd
import scipy.stats

# import matplotlib
# print(matplotlib.matplotlib_fname())
# matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import seaborn as sns


def cal_KL_divergence(data1, data2, col):
    x1 = data1[col]
    x2 = data2[col]
    if 'category' in col:
        bins = x1.unique()
        statis_result_x1 = x1.value_counts().to_dict()
        sub_count_x1 = np.array(list(map(lambda x: statis_result_x1[x] if x in statis_result_x1 else 0, bins)),
                                dtype=float)
        statis_result_x2 = x2.value_counts().to_dict()
        sub_count_x2 = np.array(list(map(lambda x: statis_result_x2[x] if x in statis_result_x2 else 0, bins)),
                                dtype=float)
    else:
        bins = np.linspace(x1.min(), x1.max(), 51)
        group_x1 = pd.cut(x1, bins=bins)
        sub_count_x1 = np.array(x1.groupby(group_x1).count(), dtype=float)
        group_x2 = pd.cut(x2, bins=bins)
        sub_count_x2 = np.array(x2.groupby(group_x2).count(), dtype=float)
    sub_prob_x1 = sub_count_x1 / sub_count_x1.sum()
    sub_prob_x2 = sub_count_x2 / sub_count_x2.sum()
    KL = scipy.stats.entropy(sub_prob_x1, sub_prob_x2)
    return KL


def plot_distribution(data1, data2, col, Title, save_name='propensity_match.png', save=False):
    KL = cal_KL_divergence(data1, data2, col)

    x1 = data1[col]
    x2 = data2[col]

    # Assign colors for each airline and the names
    colors = ['#E69F00', '#56B4E9']
    names = ['treatment', 'control']
    sns.set_style("white")
    # Make the histogram using a list of lists
    # Normalize the flights and assign colors and names
    plt.figure(figsize=(15, 6))
    if 'category' in col:
        plt.hist(x1, density=True)
        plt.hist(x2, density=True)
    else:
        plt.hist([x1, x2], bins=20, color=colors, label=names, density=True)
        sns.kdeplot(x1, color=colors[0], fill=True)
        sns.kdeplot(x2, color=colors[1], fill=True)
    # Plot formatting
    plt.legend()
    plt.xlabel('propensity logit', fontproperties='SimSun')
    plt.ylabel('density of users')
    plt.title(label=Title + 'KL: {}'.format(KL))
    # plt.show()
    if save:
        plt.savefig(save_name, dpi=250)
    else:
        plt.show()
    plt.pause(0)


if __name__ == '__main__':
    # data input path
    df = pd.read_csv('../data/user_analysis.csv')
    df.fillna('未知', inplace=True)
    statis_df = df[df.game_name == '蛋仔派对']

    data1 = statis_df[(statis_df.is_login == 1) & (statis_df.is_login_matched == 0)]
    data2 = statis_df[(statis_df.is_login == 0) & (statis_df.is_login_matched == 1)]

    # for col in statis_df.columns.tolist()[3:20]:
    #     KL = cal_KL_divergence(data1, data2, col)
    #     print("feature :{}, KL: {}".format(col, KL))

    plot_distribution(data1, data2, col='category_city_level_resident',
                      Title='propensity_match', save_name='propensity_match.png', save=True)
