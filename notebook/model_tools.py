# -*- coding: utf-8 -*-

import yaml

import numpy as np
import pandas as pd
import lightgbm as lgb

import warnings
from itertools import permutations

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier

CONTROL_GROUP = 'control'
EXP_GROUP = 'exp'


def run_uplift_model(data, feat_columns, treatment_col, label_col, params,
                     control_group_name='control',
                     model_type='tree-based'):
    if model_type == 'tree-based':
        clf = UpliftTreeClassifier(control_name=control_group_name)
        clf.fit(data[feat_columns].values,
                treatment=data[treatment_col].values,
                y=data[label_col].values)

    elif model_type == 'randomForest':
        clf = UpliftRandomForestClassifier(max_depth=params['max_depth'],
                                           min_samples_leaf=params['min_samples_leaf'],
                                           min_samples_treatment=params['min_samples_treatment'],
                                           n_reg=params['n_reg'],
                                           control_name=control_group_name,
                                           n_estimators=params['n_estimators'],
                                           random_state=1024)
        clf.fit(data[feat_columns].values,
                treatment=data[treatment_col].values,
                y=data[label_col].values)

    else:
        raise ValueError("Only support tree-based and randomForest model")

    return clf


def run_lightgbm(data, feat_columns, category_columns, label_col, params, num_boost_round=50):
    """
    Response Model
    """

    x, y = data[feat_columns], data[label_col]
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    train_data = lgb.Dataset(x_train, y_train, categorical_feature=category_columns)
    val_data = lgb.Dataset(x_test, y_test, categorical_feature=category_columns)

    # build model
    clf = lgb.train(params=params,
                    train_set=train_data,
                    num_boost_round=num_boost_round,
                    valid_sets=[train_data, val_data])
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


