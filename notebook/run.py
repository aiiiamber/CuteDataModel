# -*-coding:utf-8 -*-
# @Author: xiaolizhang


import yaml

import pandas as pd
pd.options.mode.chained_assignment = None

from notebook.psm import PropensityScoreMatching

CONTROL_GROUP = 'control'
EXP_GROUP = 'exp_content'


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
    for col in category_columns:
        feature[col] = data[col].unique().tolist()
    return feature


def print_log(data, treatment_col, label_col, return_outcome=False):
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
    if return_outcome:
        return content, outcome, t_outcome
    else:
        return content


def preprocessing_data(data, features):
    # preprocessing category columns
    for col, vocab in features.items():
        data[col] = data[col].apply(lambda x: get_index_from_list(vocab, x, -1) + 1)
    return data


def main(file_path):
    assert len(file_path) == 2, 'input file path should be 2'
    schema_path, data_path = file_path[0], file_path[1]

    # config
    print("load yaml file {} ... ".format(schema_path))
    schema = yaml.safe_load(open(schema_path, 'r', encoding='utf8'))
    config = parse_schema(schema)

    # load data
    raw_data = pd.read_csv(data_path, encoding='utf-8')
    raw_data.columns = list(schema.keys())

    treatment_col, exp_col, label_col = config['treatment'], config['exp'], config['label']
    # data distribution
    train_data = raw_data[raw_data[exp_col] == EXP_GROUP]
    train_data.fillna('未知', inplace=True)
    predicted_data = raw_data[raw_data[exp_col] == CONTROL_GROUP]
    print(">>>> Data distribution")
    print("Train data >> \n{}".format(print_log(train_data, treatment_col, label_col)))
    print("Predicted data >> \n{}".format(print_log(predicted_data, treatment_col, label_col)))

    # pearson correlation：only numerical feature
    corr_matrix = abs(train_data[[treatment_col] + config['numerical_fc']].corr())
    feat_corr = abs(corr_matrix[treatment_col]).sort_values(ascending=False)
    print("\n>>>> Feature Selection")
    print("Correlation with is treatment >> ")
    print(feat_corr[1:11])
    # feature selection
    used_feat = feat_corr[1:20].index.to_list() + config['category_fc']

    # preprocessing column
    features = parse_category_feat(train_data, config['category_fc'])
    train_data = preprocessing_data(train_data, features)

    print("\n>>>> Model Training")
    # build model
    model = PropensityScoreMatching(label_column=treatment_col,
                                    index_column=config['index'],
                                    fc_columns=used_feat,
                                    dataset=train_data,
                                    category_fc_columns=config['category_fc'],
                                    model_type='lgb')
    model.build()
    model.fit()
    # model evaluation
    model.evaluate()

    # predict
    predicted_data = preprocessing_data(predicted_data, features)
    control_data = model.predict_propensity_score(predicted_data, saved_columns=[label_col])
    exp_content, exp_outcome, exp_t_outcome = print_log(train_data, treatment_col, label_col, return_outcome=True)
    control_content, control_outcome, control_t_outcome = print_log(control_data, 'binary_predicted_label', label_col,
                                                                    return_outcome=True)
    print("Exp data >> \n{}".format(exp_content))
    print("Control data >> \n{}".format(control_content))
    print('Treatment Effect: ATE: {ae: .4%}, ATT: {at:.4%}'.format(ae=exp_outcome / control_outcome - 1,
                                                                   at=exp_t_outcome / control_t_outcome - 1))


if __name__ == '__main__':
    inp = '../config/schema.yaml,../data/train.csv'
    file_path = inp.split(",")

    main(file_path)
