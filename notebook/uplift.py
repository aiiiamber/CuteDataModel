# -*- coding: utf-8 -*-

import yaml

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
from causalml.inference.tree import uplift_tree_string, uplift_tree_plot
from causalml.metrics.visualize import auuc_score, get_cumlift

from sklearn.model_selection import train_test_split

from notebook.tools import parse_schema
from notebook.test import ks_test

CONTROL_GROUP = 'control'
EXP_GROUP = 'exp'


def evaluate(df, outcome_col, treatment_col, plot=False):
    lift = get_cumlift(df, outcome_col=outcome_col, treatment_col=treatment_col)
    gain = lift.mul(lift.index.values, axis=0)

    # format cumalative gains：todo：need fixed
    grouped = df.pivot_table(values=outcome_col, index=[treatment_col], aggfunc=[np.mean, np.size], margins=True)
    control_conversion_ucnt = grouped.loc[0, 'mean'].values[0] * grouped.loc[0, 'size'].values[0]
    exp_conversion_ucnt = grouped.loc[1, 'mean'].values[0] * grouped.loc[1, 'size'].values[0]
    rct_lift_cnt = exp_conversion_ucnt - control_conversion_ucnt
    modify_factor = rct_lift_cnt / gain.max().min()
    greatest_gain = gain.max() * modify_factor / control_conversion_ucnt

    gain = gain * modify_factor / control_conversion_ucnt

    # metric
    auuc_metric = auuc_score(df, outcome_col=outcome_col, treatment_col=treatment_col)
    print("AUUC metric: \n", auuc_metric)
    title_name = '{}'.format(', '.join(["{}:{:.2%}".format(k, v) for k, v in greatest_gain.items()]))
    print(title_name)

    if plot:
        # plot gini curve
        gain.plot()
        plt.figure(figsize=(20, 16))
        # gain = gain.div(np.abs(gain.iloc[-1, :]))  # 纵坐标归一化缩放
        plt.xlabel('Cumulative of people targeted')
        plt.ylabel('Cumulative incremental gains')
        plt.legend()
        gain.plot()
        plt.savefig('../tmp/pic/gini_curve.png')


def run_lightgbm(data, feat_columns, category_columns, label_col, num_boost_round=50):
    """
    Response Model
    :param data:
    :param feat_columns:
    :param category_columns:
    :param label_col:
    :param num_boost_round:
    :return:
    """

    x, y = data[feat_columns], data[label_col]
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    train_data = lgb.Dataset(x_train, y_train, categorical_feature=category_columns)
    val_data = lgb.Dataset(x_test, y_test, categorical_feature=category_columns)

    # load params
    params = yaml.safe_load(open('../config/models.yaml', 'r', encoding='utf8'))['lightgbm']
    # build model
    clf = lgb.train(params=params,
                    train_set=train_data,
                    num_boost_round=num_boost_round,
                    valid_sets=[train_data, val_data])
    return clf


def run_uplift_model(data, feat_columns, treatment_col, label_col, control_group_name='control',
                     model_type='tree-based'):
    if model_type == 'tree-based':
        clf = UpliftTreeClassifier(control_name=control_group_name)
        clf.fit(data[feat_columns].values,
                treatment=data[treatment_col].values,
                y=data[label_col].values)
    elif model_type == 'randomForest':
        clf = UpliftRandomForestClassifier(max_depth=4,
                                           min_samples_leaf=200,
                                           min_samples_treatment=20,
                                           n_reg=10,
                                           evaluationFunction='KL',
                                           control_name=control_group_name)
        clf.fit(data[feat_columns].values,
                treatment=data[treatment_col].values,
                y=data[label_col].values)
    else:
        raise ValueError("Only support tree-based and randomForest model")

    return clf


def generate_evaluation_result(data, feat_columns, label_col, uplift_model, response_model,
                               uplift_model_type='tree-based'):
    df_test = data.reset_index(drop=True)
    # uplift model
    y_pred = uplift_model.predict(df_test[feat_columns].values)
    if uplift_model_type == 'tree-based':
        y_pred = y_pred[:, 1] - y_pred[:, 0]
    df_res = pd.DataFrame(y_pred, columns=[uplift_model_type])
    # response model
    y_prod = response_model.predict(df_test[feat_columns])
    df_res['response_model'] = y_prod
    df_res['t'] = df_test['is_exp_treatment']
    df_res['y'] = df_test[label_col]
    # store result
    df_test['causalML'] = y_pred
    df_test['response_model'] = y_prod
    return df_res, df_test


def main(input, model_type='randomForest', dataset=None, offline=True, store_res=False):
    file_path = input.split(",")
    if offline:
        assert len(file_path) == 3, 'input file path should be 2'
        schema_path, data_path = file_path[0], file_path[1]
    else:
        schema_path = file_path[0]

    # config
    print("load yaml file {} ... ".format(schema_path))
    schema = yaml.safe_load(open(schema_path, 'r', encoding='utf8'))
    config = parse_schema(schema)

    # load data
    if offline:
        raw_data = pd.read_csv(data_path, encoding='utf-8')
    else:
        raw_data = dataset
    raw_data.columns = list(schema.keys())

    # Step1：数据预处理
    raw_data.fillna('0', inplace=True)
    raw_data[config['index']] = raw_data[config['index']].astype(str)
    raw_data[config['numerical_fc']] = raw_data[config['numerical_fc']].apply(pd.to_numeric)
    # generate config
    feat_names = config['numerical_fc']
    treatment_col, exp_col, label_col = config['treatment'], config['exp'], config['label']
    raw_data['is_exp_treatment'] = raw_data[exp_col].apply(lambda x: 1 if x == EXP_GROUP else 0)
    # df = raw_data.sample(frac=1).reset_index(drop=True) # random
    df = raw_data
    print(df.pivot_table(values=label_col, index=[exp_col, treatment_col], aggfunc=[np.mean, np.size]))
    # Split data to training and testing samples for model validation (next section)
    train_data, test_data = train_test_split(df, test_size=0.3, random_state=1024)
    # ks_test(train_data, test_data, feat_names + [label_col, 'is_exp_treatment'])
    print('Data distribution ...')
    print(train_data.pivot_table(values=label_col, index=[exp_col], aggfunc=[np.mean, np.size]))
    print(test_data.pivot_table(values=label_col, index=[exp_col], aggfunc=[np.mean, np.size]))

    # Step2：训练模型
    # training uplift model
    print('Uplift Model training ...')
    uplift_model = run_uplift_model(train_data,
                                    feat_columns=feat_names,
                                    treatment_col=exp_col,
                                    label_col=label_col,
                                    model_type=model_type)
    # training response model
    print('Response Model training ...')
    response_model = run_lightgbm(data=train_data,
                                  feat_columns=feat_names,
                                  category_columns=[],
                                  label_col=label_col)

    # Only for model_type tree-based
    if model_type == 'tree-based' and store_res:
        # Plot uplift tree
        graph = uplift_tree_plot(uplift_model.fitted_uplift_tree, feat_names)
        graph.write_png("../tmp/pic/dtr.png")

    # Step3：评估模型结果
    # Prediction and Evaluation
    train_res, df_train = generate_evaluation_result(train_data,
                                                     feat_columns=feat_names,
                                                     label_col=label_col,
                                                     uplift_model=uplift_model,
                                                     response_model=response_model,
                                                     uplift_model_type=model_type)
    evaluate(train_res, outcome_col='y', treatment_col='t', plot=True)
    test_res, df_test = generate_evaluation_result(test_data,
                                                   feat_columns=feat_names,
                                                   label_col=label_col,
                                                   uplift_model=uplift_model,
                                                   response_model=response_model,
                                                   uplift_model_type=model_type)
    evaluate(test_res, outcome_col='y', treatment_col='t', plot=False)

    # store result
    df_train.to_csv('../data/res/res.csv', encoding='utf-8', index=False)


if __name__ == '__main__':
    # data input path
    input = '../config/schema-new.yaml,../data/train/train-0910.csv, ../data/match.csv'

    # model training config
    main(input, model_type='randomForest', store_res=True)
