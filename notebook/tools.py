# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from itertools import combinations, permutations


def get_index_from_list(lst, ele, default_value=None):
    """Get the index of an element in a list; return None if does not exist.

    :param lst: a list.
    :param ele: an element whose index is the target.
    :param default_value: if element not in index, return default value
    :return: an int indicates the index of the `ele` in `lst`; returns None if does not exist.
    """

    if not isinstance(lst, list):
        return None

    try:
        return lst.index(ele)
    except ValueError:
        return default_value


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


def parse_category_feat(data, category_columns):
    feature = dict()
    # print(category_columns)
    for col in category_columns:
        feature[col] = data[col].unique().tolist()
    return feature


def preprocessing_data(data, features):
    # preprocessing category columns
    for col, vocab in features.items():
        data[col] = data[col].apply(lambda x: get_index_from_list(vocab, x, -1) + 1)
    return data


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