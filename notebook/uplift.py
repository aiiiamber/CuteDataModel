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

CONTROL_GROUP = 'control'
EXP_GROUP = 'exp_content'


def evaluate(df, outcome_col, treatment_col):
    lift = get_cumlift(df, outcome_col=outcome_col, treatment_col=treatment_col)
    gain = lift.mul(lift.index.values, axis=0)
    gain.plot()
    plt.figure(figsize=(10, 8))
    # gain = gain.div(np.abs(gain.iloc[-1, :]))  # 纵坐标归一化缩放
    gain.plot()
    plt.title()
    plt.savefig('../tmp/pic/gini_curve.png')

    print(auuc_score(df))


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


def main(input, model_type='randomForest', dataset=None, offline=True):
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
    raw_data.fillna('未知', inplace=True)
    raw_data[config['index']] = raw_data[config['index']].astype(str)
    raw_data[config['numerical_fc']] = raw_data[config['numerical_fc']].apply(pd.to_numeric)

    feat_names = config['numerical_fc']
    treatment_col, exp_col, label_col = config['treatment'], config['exp'], config['label']
    raw_data['is_exp_treatment'] = raw_data[exp_col].apply(lambda x: 1 if x == EXP_GROUP else 0)
    df = raw_data.sample(frac=1).reset_index(drop=True)
    print(df.pivot_table(values=label_col, index=[exp_col, treatment_col], aggfunc=[np.mean, np.size], margins=True))

    print('Model training ...')
    # Split data to training and testing samples for model validation (next section)
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=1024)
    print('Data distribution ...')
    print(
        df_train.pivot_table(values=label_col, index=[exp_col, 'is_exp_treatment'], aggfunc=[np.mean, np.size],
                             margins=True))
    print(
        df_test.pivot_table(values=label_col, index=[exp_col, 'is_exp_treatment'], aggfunc=[np.mean, np.size],
                            margins=True))

    # Train uplift model
    uplift_model = UpliftRandomForestClassifier(max_depth=4,
                                                min_samples_leaf=200,
                                                min_samples_treatment=20,
                                                n_reg=10,
                                                evaluationFunction='KL',
                                                control_name='control')

    uplift_model.fit(df_train[feat_names].values, treatment=df_train[exp_col].values, y=df_train[label_col].values)

    # Only for UpliftTreeClassifier mode
    if model_type == 'tree-based':
        # Print uplift tree as a string
        result = uplift_tree_string(uplift_model.fitted_uplift_tree, feat_names)
        # Plot uplift tree
        graph = uplift_tree_plot(uplift_model.fitted_uplift_tree, feat_names)
        graph.write_png("dtr.png")

    # Train response model
    response_model = run_lightgbm(data=df_train,
                                  feat_columns=feat_names,
                                  category_columns=[],
                                  label_col=label_col)

    # Prediction and Evaluation
    df_test = df_train
    df_test.reset_index(drop=True, inplace=True)
    # uplift model
    y_pred = uplift_model.predict(df_test[feat_names].values)
    df_res = pd.DataFrame(y_pred)
    df_res.columns = ['causalML']
    # response model
    y_prod = response_model.predict(df_test[feat_names])
    df_res['response_model'] = y_prod
    df_res['w'] = df_test['is_exp_treatment']
    df_res['y'] = df_test[label_col]
    print(df_res.pivot_table(values='y', index='w', aggfunc=[np.mean, np.size], margins=True))

    evaluate(df_res, outcome_col='y', treatment_col='w')


if __name__ == '__main__':
    # data input path
    input = '../config/schema.yaml,../data/train.csv, ../data/match.csv'

    # model training config
    main(input)
