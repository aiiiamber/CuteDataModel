# -*- coding: utf-8 -*-

import yaml

import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.model_selection import train_test_split

import matplotlib

matplotlib.use('TkAgg')
import seaborn as sns

sns.set(rc={'figure.figsize': (10, 8)}, font_scale=1.3)
import matplotlib.pyplot as plt

CONTROL_GROUP = 'control'
EXP_GROUP = 'exp_content'


def parse_schema(schema):
    """
    Parse config
    :param schema: dictionary
    :return:
    """
    config = dict()
    for key, value in schema.items():
        # outcome
        if value == 'label':
            config['label'] = key
        # uid
        if value == 'index':
            config['index'] = key
        # experiment group
        if value == 'exp':
            config['exp'] = key
        # treatment label
        if value == 'treatment':
            config['treatment'] = key

        # feature column
        if value == 'used':
            fc_columns, numerical_fc_columns, category_fc_columns = [], [], []
            if 'fc' in config:
                fc_columns.extend(config['fc'])
            if 'category' not in key:
                fc_columns.append(key)
            config['fc'] = fc_columns

            if 'category' not in key:
                if 'numerical_fc' in config:
                    numerical_fc_columns.extend(config['numerical_fc'])
                numerical_fc_columns.append(key)
                config['numerical_fc'] = numerical_fc_columns
            else:
                if 'category_fc' in config:
                    category_fc_columns.extend(config['category_fc'])
                category_fc_columns.append(key)
                config['category_fc'] = category_fc_columns

    return config


def run_model(train_data, feature_columns, category_columns, label, num_boost_round=100):
    print('split training data ...')
    x, y = train_data[feature_columns], train_data[label]
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    train_data = lgb.Dataset(x_train, y_train)
    val_data = lgb.Dataset(x_test, y_test)
    # train_data = lgb.Dataset(x_train, y_train, categorical_feature=category_columns)
    # val_data = lgb.Dataset(x_test, y_test, categorical_feature=category_columns)

    # build model
    print('training model ...')
    params = {
        'num_leaves': 32,
        'max_depth': 8,
        'min_data_in_leaf': 20,
        'min_child_samples': 20,
        'objective': 'binary',
        'learning_rate': 0.1,
        'boosting': 'gbdt',
        'feature_fraction': 0.8,
        'bagging_freq': 0,
        'bagging_fraction': 0.8,
        'bagging_seed': 23,
        'metric': 'auc',
        'lambda_l1': 0.2,
        'nthread': 4,
        'verbose': -1
    }
    clf = lgb.train(params=params,
                    train_set=train_data,
                    num_boost_round=num_boost_round,
                    valid_sets=[train_data, val_data])
    return clf


def main(input, dataset=None, offline=True):
    file_path = input.split(",")
    if offline:
        assert len(file_path) == 2, 'input file path should be 2'
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
    raw_data.fillna('未知', inplace=True)
    raw_data[config['index']] = raw_data[config['index']].astype(str)
    raw_data[config['numerical_fc']] = raw_data[config['numerical_fc']].apply(pd.to_numeric)

    treatment_col, exp_col, label_col = config['treatment'], config['exp'], config['label']
    # data distribution
    treatment_data = raw_data[raw_data[exp_col] == EXP_GROUP]
    control_data = raw_data[raw_data[exp_col] == CONTROL_GROUP]

    treatment_model = run_model(treatment_data,
                                feature_columns=config['fc'],
                                category_columns=config['category_fc'],
                                label=label_col,
                                num_boost_round=100)
    control_model = run_model(control_data,
                              feature_columns=config['fc'],
                              category_columns=config['category_fc'],
                              label=label_col,
                              num_boost_round=100)

    treatment_t_predict = treatment_model.predict(treatment_data[config['fc']])
    untreatment_t_predict = control_model.predict(treatment_data[config['fc']])
    treatment_data['uplift'] = list(treatment_t_predict - untreatment_t_predict)

    treatment_c_predict = treatment_model.predict(control_data[config['fc']])
    untreatment_c_predict = control_model.predict(control_data[config['fc']])
    control_data['uplift'] = list(treatment_c_predict - untreatment_c_predict)

    bins = np.linspace(treatment_data.uplift.min(), treatment_data.uplift.max(), 10)
    treatment_data['grouped'] = pd.cut(treatment_data['uplift'], bins=bins)
    res_t = treatment_data.groupby('grouped')[['uplift', 'is_login']].mean()
    control_data['grouped'] = pd.cut(treatment_data['uplift'], bins=bins)
    res_c = control_data.groupby('grouped')['is_login'].mean().to_dict()



    # plt.figure(figsize=(15, 6))
    # plt.plot(x, res.uplift)
    # plt.plot(x, res.is_login)
    # plt.show()

    return None


if __name__ == '__main__':
    # data input path
    input = '../config/schema.yaml,../data/train.csv'

    # model training config
    main(input)

