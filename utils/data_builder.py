# -*-coding:utf-8 -*-
# @Author: xiaolizhang

import numpy as np
import pandas as pd


def dict_to_yaml(map_dict):
    content = ''

    for key, key_2_value in map_dict.items():
        content += '{}：{}\n'.format('feature_name', key)
        for k, v in key_2_value.items():
            content += '    {}：{}\n'.format(k, v)
        content += '\n'
    return content


def preprocessing_data(conf):
    data = []
    # iterable load data
    for filename in conf.args.data_input.split(","):
        f = open(filename)
        lines = f.readlines()
        for i, l in enumerate(lines):
            line_info = l.strip('\n').replace('"', '').split(',')
            if len(line_info) == len(conf.column_names):
                data.append(line_info)
    dataset = pd.DataFrame(np.array(data), columns=conf.column_names)

    # numerical column
    num_columns = conf.numerical_columns + [conf.label_column]
    dataset[num_columns] = dataset[num_columns].astype('float32')

    # generate category mapping info
    if conf.has_category:
        category_map = dict()
        for col in conf.category_columns:
            print("process category column: {}".format(col))
            unique_keys = sorted(dataset[col].unique())
            key_2_value = dict(zip(unique_keys, list(range(len(unique_keys)))))
            category_map[col] = key_2_value
            dataset[col] = dataset[col].map(lambda x: key_2_value[x])
        yaml_file = './config/feature.yaml'
        print('writing {}...'.format(yaml_file))
        with open(yaml_file, 'w', encoding='utf8') as f:
            f.write(dict_to_yaml(category_map))

    # balance positive and negative sample number
    if conf.balanced:
        positive = dataset[dataset[conf.label_column] == 1]
        negative = dataset[dataset[conf.label_column] == 0]
        if positive.shape[0] > negative.shape[0]:
            positive = positive.sample(n=negative.shape[0])
        else:
            negative = negative.sample(n=positive.shape[0])
        dataset = pd.concat([positive, negative])
        dataset.sample(frac=1).reset_index(drop=True)  # shuffle

    return dataset