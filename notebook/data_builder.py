# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from itertools import permutations


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


def feature_selection(data, feature_columns, label_column):
    corr_matrix = data[feature_columns + [label_column]].corr()
    filter_columns = []

    for col_i, col_j in permutations(feature_columns, 2):
        if col_i in filter_columns:
            continue

        corr = corr_matrix[col_i][col_j]

        if 0.7 <= abs(corr) <= 1:
            if abs(corr_matrix[col_i][label_column]) >= abs(corr_matrix[col_j][label_column]):
                filter_columns.append(col_j)
            else:
                filter_columns.append(col_i)

    selected_columns = list(set(feature_columns) - set(filter_columns))

    return selected_columns


def dataset_builder(df, categorical_columns, numerical_columns, key_column, if_format):
    feature_map = dict()
    final_columns = []
    for col in categorical_columns:
        feat, feat_config = _process_category_cols(df[col])
        df[col] = feat
        final_columns.append(col)
        feature_map[col] = feat_config

    # numerical feature, min_max format feature scale
    if if_format:
        for col in numerical_columns:
            if df[col].unique().shape[0] > 100:
                num_feat, feat_config = _process_numerical_cols(df[col], df[key_column])
                df['format_' + col] = num_feat
                final_columns.append('format_' + col)
                df[col] = feat_config
            else:
                final_columns.append(col)
    else:
        final_columns.extend(numerical_columns)

    return df, feature_map, final_columns


def _process_category_cols(feat):
    unique_key = feat.unique().tolist()
    feat_map = {k: i for i, k in enumerate(unique_key)}
    feat = feat.map(feat_map)
    return feat, feat_map


def _process_numerical_cols(feat, key):
    df = pd.DataFrame(np.array([feat.values, key.values]).T, columns=['v', 'k'])
    grouped = df.groupby('k')['v'].max().to_dict()

    min_val = 0
    feat = [(v - min_val) / (grouped[k] - min_val) for v, k in zip(feat, key)]
    return feat, grouped
