# -*- coding: utf-8 -*-

import yaml

import numpy as np
import pandas as pd

from causalml.dataset import make_uplift_classification
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
from causalml.inference.tree import uplift_tree_string, uplift_tree_plot

import lightgbm as lgb
from sklearn.model_selection import train_test_split

import matplotlib

matplotlib.use('TkAgg')
import seaborn as sns

sns.set(rc={'figure.figsize': (10, 8)}, font_scale=1.3)
# from IPython.display import Image
from notebook.tools import parse_schema, parse_category_feat, preprocessing_data

CONTROL_GROUP = 'control'
EXP_GROUP = 'exp_content'


def main(input, dataset=None, offline=True):
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
    df = raw_data[raw_data[exp_col].isin([CONTROL_GROUP, EXP_GROUP])]

    # Split data to training and testing samples for model validation (next section)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=111)

    # Train uplift tree
    uplift_model = UpliftTreeClassifier(max_depth=4, min_samples_leaf=200, min_samples_treatment=50, n_reg=100,
                                        evaluationFunction='KL', control_name='control')

    uplift_model.fit(df_train[feat_names].values,
                     treatment=df_train[exp_col].values,
                     y=df_train[label_col].values)

    # Print uplift tree as a string
    result = uplift_tree_string(uplift_model.fitted_uplift_tree, feat_names)

    # Plot uplift tree
    graph = uplift_tree_plot(uplift_model.fitted_uplift_tree, feat_names)
    graph.write_png("dtr.png")


if __name__ == '__main__':
    # data input path
    input = '../config/schema.yaml,../data/train.csv, ../data/match.csv'

    # model training config
    main(input)
