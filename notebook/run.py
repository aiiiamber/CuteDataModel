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
from scipy.stats import gaussian_kde

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
    # print(category_columns)
    for col in category_columns:
        feature[col] = data[col].unique().tolist()
    return feature


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
    if return_outcome:
        return content, outcome, t_outcome
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
        pass


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
    plt.hist([x1, x2], color=colors, label=names, density=True)
    plt.plot(x1, density(xs))
    # Plot formatting
    plt.legend()
    plt.xlabel('propensity logit')
    plt.ylabel('density of users')
    plt.title(label=Title)
    if save:
        plt.savefig(save_name, dpi=250)
        plt.pause(0)
    else:
        pass


def preprocessing_data(data, features):
    # preprocessing category columns
    for col, vocab in features.items():
        data[col] = data[col].apply(lambda x: get_index_from_list(vocab, x, -1) + 1)
    return data


def main(input, model_config, dataset=None, offline=True):
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
    train_data = raw_data[raw_data[exp_col] == EXP_GROUP]
    control_data = raw_data[raw_data[exp_col] == CONTROL_GROUP]
    print(">>>> Data distribution")
    print("Train data >> \n{}".format(generate_distribution(train_data, treatment_col, label_col)))
    print("Control data >> \n{}".format(generate_distribution(control_data, treatment_col, label_col)))

    # pearson correlation：only numerical feature
    corr_matrix = abs(train_data[[treatment_col] + config['numerical_fc']].corr())
    feat_corr = abs(corr_matrix[treatment_col]).sort_values(ascending=False)

    corr_y_matrix = abs(train_data[[label_col] + config['numerical_fc']].corr())
    feat_corr_y = abs(corr_y_matrix[label_col]).sort_values(ascending=False).to_dict()

    print("\n>>>> Feature Selection")
    print("Selected features correlation with is treatment >> ")
    # print(feat_corr[1:11])
    # feature selection
    used_feat = []
    used_feat.extend(config['category_fc'])
    filter_col = []
    for k, v in feat_corr[1:].to_dict().items():
        if k in filter_col:
            continue

        corr_y = feat_corr_y[k]
        if v < 0.1 and corr_y < 0.1:
            print("skipped feature {}".format(k))
            filter_col.append(k)
        else:
            print("feature: {}, correlation with treatment: {c_t: .4f}, with outcome: {c_o: .4f}".format(k,
                                                                                                         c_t=v,
                                                                                                         c_o=corr_y))
            used_feat.append(k)
            # filter high correlation feature
            cols = corr_matrix[(corr_matrix[k] > 0.6) & (corr_matrix[k] < 1)][k].index.to_list()
            filter_col.extend(cols)
    print("used feature number: {}".format(len(used_feat)))

    # preprocessing data
    features = parse_category_feat(train_data, config['category_fc'])
    train_data = preprocessing_data(train_data, features)
    control_data = preprocessing_data(control_data, features)

    print("\n>>>> Build Model")
    # build model
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
    plot_match(df_after, treatment_col, Title='Propensity Scores Before Matching',
               save_name='propensity_match_before.png', save=True)
    plot_match(df_after[df_after['user_id'].isin(pairs)], treatment_col, Title='Propensity Scores After Matching',
               save_name='propensity_match_after.png', save=True)
    plot_effect_size(df_before, df_after[df_after['user_id'].isin(pairs)],
                     treatment=treatment_col,
                     vars=used_feat,
                     save=True)

    # treatment effect
    control_data = model.predict_propensity_score(control_data, saved_columns=[label_col] + used_feat)
    exp_content, exp_outcome, exp_t_outcome = generate_distribution(train_data, treatment_col, label_col,
                                                                    return_outcome=True)
    control_content, control_outcome, control_t_outcome = generate_distribution(control_data, 'binary_predicted_label',
                                                                                label_col,
                                                                                return_outcome=True)
    print("Exp data >> \n{}".format(exp_content))
    print("Control data >> \n{}".format(control_content))
    print('Treatment Effect: ATE: {ae: .4%}, ATT: {at:.4%}'.format(ae=exp_outcome / control_outcome - 1,
                                                                   at=exp_t_outcome / control_t_outcome - 1))


if __name__ == '__main__':
    # data input path
    input = '../config/schema.yaml,../data/train.csv'
    # model config
    model_config = {
            'model_type': 'lgb',
            'num_boost_round': 100
        }

    # model training config
    main(input, model_config)
