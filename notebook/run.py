# -*-coding:utf-8 -*-
# @Author: xiaolizhang


import yaml
import math

import pandas as pd

pd.options.mode.chained_assignment = None
import matplotlib

matplotlib.use('TkAgg')
import seaborn as sns

sns.set(rc={'figure.figsize': (10, 8)}, font_scale=1.3)
import matplotlib.pyplot as plt

from notebook.psm import PropensityScoreMatching
from notebook.tools import preprocessing_data, parse_schema, parse_category_feat
from notebook.model_tools import *

CONTROL_GROUP = 'control'
EXP_GROUP = 'exp_content'


def generate_distribution(data, treatment_col, label_col, return_outcome=False):
    data['binary_treatment'] = data[treatment_col].apply(lambda x: 1 if x > 0 else 0)
    sample_cnt = data.shape[0]
    positive_cnt = data['binary_treatment'].sum()
    content = "positive rate: {r:.2%}, sample counts: {cnt} \n".format(r=positive_cnt / sample_cnt, cnt=sample_cnt)

    data['binary_label'] = data[label_col].apply(lambda x: 1 if x > 0 else 0)
    outcome = data.binary_label.mean()
    t_outcome = data[data['binary_treatment'] == 1]['binary_label'].mean()
    ut_outcome = data[data['binary_treatment'] == 0]['binary_label'].mean()
    content += "outcome:{r:.2%}, treatment outcome: {r1:.2%}, untreatment outcome: {r2:.2%}".format(r=outcome,
                                                                                                    r1=t_outcome,
                                                                                                    r2=ut_outcome)
    res = [positive_cnt / sample_cnt, outcome, t_outcome]
    if return_outcome:
        return content, res
    else:
        return content


def plot_effect_size(df_before, df_after, treatment, vars, save=False):
    title = 'Standardized Mean differences accross covariates before and after matching'
    before_color, after_color = '#FCB754', '#3EC8FB'

    def cohenD(df, treatment, metricName):
        treated_metric = df[df[treatment] == 1][metricName]
        untreated_metric = df[df[treatment] == 0][metricName]
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

    data = []
    for cl in vars:
        if 'category' not in cl:
            data.append([cl, 'before', cohenD(
                df_before, treatment, cl)])
            data.append([cl, 'after', cohenD(
                df_after, treatment, cl)])
    effect_size = pd.DataFrame(data, columns=['Variable', 'matching', 'Effect Size'])
    sns.set_style("white")
    sns_plot = sns.barplot(data=effect_size, y='Variable', x='Effect Size', hue='matching', palette=[
        before_color, after_color], orient='h')
    sns_plot.set(title=title)
    if save:
        sns_plot.figure.savefig(
            'effect_size.png', dpi=250, bbox_inches="tight")
    else:
        return effect_size


def plot_match(df, treatment, Title, save_name='propensity_match.png', save=False):
    # data process
    dftreat = df[df[treatment] == 1]
    dfcontrol = df[df[treatment] == 0]
    matched_entity = 'propensity_logit'
    x1 = dftreat[matched_entity]
    x2 = dfcontrol[matched_entity]
    # Assign colors for each airline and the names
    colors = ['#E69F00', '#56B4E9']
    names = ['treatment', 'control']
    sns.set_style("white")
    # Make the histogram using a list of lists
    # Normalize the flights and assign colors and names
    plt.figure(figsize=(15, 6))
    plt.hist([x1, x2], bins=20, color=colors, label=names, density=True)
    sns.kdeplot(x1, color=colors[0], fill=True)
    sns.kdeplot(x2, color=colors[1], fill=True)
    # Plot formatting
    plt.legend()
    plt.xlabel('propensity logit')
    plt.ylabel('density of users')
    plt.title(label=Title)
    # plt.show()
    if save:
        plt.savefig(save_name, dpi=250)
        plt.pause(0)
    else:
        return


def main(input, model_config, dataset=None, offline=True):
    file_path = input.split(",")
    if offline:
        assert len(file_path) == 2, 'input file path should be 2'
        schema_path, data_path = file_path[0], file_path[1]
    else:
        schema_path = file_path[0]

    # config：读取基础配置
    print("load yaml file {} ... ".format(schema_path))
    schema = yaml.safe_load(open(schema_path, 'r', encoding='utf8'))
    config = parse_schema(schema)
    treatment_col, exp_col, label_col = config['treatment'], config['exp'], config['label']
    fc_cols, num_fc_cols, categorical_fc_cols = config['fc'], config['numerical_fc'], config['category_fc']

    # load data：加载数据
    if offline:
        raw_data = pd.read_csv(data_path, encoding='utf-8')
    else:
        raw_data = dataset
    raw_data.columns = list(schema.keys())
    raw_data.fillna(0, inplace=True)  # 填充Nan
    raw_data[config['index']] = raw_data[config['index']].astype(str)
    raw_data[num_fc_cols] = raw_data[num_fc_cols].apply(pd.to_numeric)

    print("\n>>>> Feature Selection")
    used_feat = feature_selection(raw_data, num_fc_cols, label_col) if model_config['feature_selection'] else fc_cols
    print("used feature size: {}".format(len(used_feat)))

    # data distribution
    print(">>>> Data distribution")
    print(raw_data.pivot_table(values=label_col,
                               index=['game_tag_name', exp_col, treatment_col],
                               aggfunc=[np.mean, np.size]))
    train_data = raw_data[raw_data[exp_col] == EXP_GROUP]
    control_data = raw_data[raw_data[exp_col] == CONTROL_GROUP]

    # preprocessing data
    features = parse_category_feat(train_data, config['category_fc'])
    train_data = preprocessing_data(train_data, features)
    control_data = preprocessing_data(control_data, features)

    print("\n>>>> Build PSM")
    # STEP1：PSM匹配
    model = PropensityScoreMatching(label_column=treatment_col,
                                    index_column=config['index'],
                                    fc_columns=used_feat,
                                    dataset=train_data,
                                    control_data=control_data,
                                    model_config=model_config,
                                    category_fc_columns=config['category_fc'])
    model.build()
    model.fit()
    # model evaluation
    model.evaluate()

    # balanced test
    df_treatment, df_before_control = train_data[train_data[treatment_col] == 1], control_data
    df_before_control[treatment_col] = 0
    df_before = pd.concat([df_treatment, df_before_control], ignore_index=True)
    df_after, df_matched = model.predict()
    pairs = df_matched.matched_ID.unique().tolist() + df_matched.user_id.unique().tolist()
    # plot_match(df_after, treatment_col, Title='Propensity Scores Before Matching',
    #            save_name='propensity_match_before.png', save=True)
    # plot_match(df_after[df_after['user_id'].isin(pairs)], treatment_col, Title='Propensity Scores After Matching',
    #            save_name='propensity_match_after.png', save=True)
    effect_size = plot_effect_size(df_before, df_after[df_after['user_id'].isin(pairs)],
                                   treatment=treatment_col,
                                   vars=used_feat,
                                   save=False)
    print(effect_size[effect_size['Variable'] == model_config['stratification_feature']])

    # treatment effect
    control_data = model.predict_propensity_score(control_data, saved_columns=[label_col] + used_feat)
    exp_content, exp_res = generate_distribution(train_data, treatment_col, label_col, return_outcome=True)
    control_content, control_res = generate_distribution(control_data,
                                                         'binary_predicted_label',
                                                         label_col,
                                                         return_outcome=True)
    print("Exp data >> \n{}".format(exp_content))
    print("Control data >> \n{}".format(control_content))
    ate, att = exp_res[1] / control_res[1] - 1, exp_res[2] / control_res[2] - 1
    print('Treatment Effect: ATE: {ae: .4%}, ATT: {at:.4%}'.format(ae=ate, at=att))
    att_before = control_before_res[2] / exp_res[2] - 1
    res = [ate, att, exp_res[0], control_before_res[0], exp_res[0], control_res[0],
           exp_res[2], control_before_res[2], att_before, exp_res[2], control_res[2]]

    saved_columns = used_feat + [treatment_col, exp_col, label_col, config['index']]
    matched_pair = df_matched[['user_id', 'matched_ID']]

    matched_res = pd.merge(matched_pair, raw_data[saved_columns], how='inner', on='user_id')
    matched_res = pd.merge(matched_res, raw_data[saved_columns], how='inner',
                           left_on='matched_ID', right_on='user_id', suffixes=('', '_matched'))
    return res, matched_res


if __name__ == '__main__':
    # data input path
    input = '../config/schema-model.yaml,../data/train.csv'
    # model config
    model_config = {
        'model_type': 'lgb',  # lgb, logit
        'num_boost_round': 50,  # only need under lgb model_type
        'match_type': 'knn',  # stratification_match, knn
        'stratification_feature': 'recent_30d_active_cnt',
        'feature_selection': False
    }

    # model training config
    res, matched_res = main(input, model_config)
    matched_res.to_csv('../data/match.csv')
    print(res)
