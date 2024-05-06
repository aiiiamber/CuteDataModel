# -*-coding:utf-8 -*-
# @Author: xiaolizhang


import yaml

import pandas as pd
import lightgbm as lgb

from .psm import PropensityScoreMatching
from sklearn.model_selection import train_test_split


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


def main(file_path):
    assert len(file_path) == 2, 'input file path should be 2'
    schema_path, data_path = file_path[0], file_path[1]

    # config
    print("load yaml file {} ... ".format(schema_path))
    schema = yaml.safe_load(open(schema_path, 'r', encoding='utf8'))
    config = parse_schema(schema)

    # load data
    raw_data = pd.read_csv(data_path,  encoding='utf-8')
    raw_data.columns = list(schema.keys())

    # pearson correlation：only numerical feature
    treatment_col = config['treatment']
    num_corr = abs(raw_data[[treatment_col] + config['numerical_fc']].corr()[treatment_col]).sort_values(ascending=False)
    print("correlation with is treatment >> ")
    print(num_corr[1:11])

    # build model
    model = PropensityScoreMatching(label_column=treatment_col,
                                    index_column=config['index'],
                                    fc_columns=config['numerical_fc'],
                                    dataset=raw_data,
                                    model_type='lgb')
    model.build()
    model.fit()
    model.evaluate()


if __name__ == '__main__':
    inp = '../config/schema.yaml,../data/model/train.csv'
    file_path = inp.split(",")

    main(file_path)
