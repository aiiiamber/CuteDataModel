# -*-coding:utf-8 -*-
# @Author: xiaolizhang
import os

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


def preprocessing_data(conf, fc):
    data = []
    # iterable load data
    for filename in conf.args.data_input.split(","):
        f = open(filename)
        lines = f.readlines()
        for i, l in enumerate(lines):
            line_info = l.strip('\n').replace('"', '').split(',')
            if len(line_info) == len(fc.column_names):
                data.append(line_info)
    dataset = pd.DataFrame(np.array(data), columns=fc.column_names)

    # numerical column
    num_columns = fc.numerical_columns + [fc.label_column]
    dataset[num_columns] = dataset[num_columns].astype('float32')

    # generate category mapping info
    if fc.has_category:
        category_map = dict()
        for col in fc.category_columns:
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
        positive = dataset[dataset[fc.label_column] == 1]
        negative = dataset[dataset[fc.label_column] == 0]
        if positive.shape[0] > negative.shape[0]:
            positive = positive.sample(n=negative.shape[0])
        else:
            negative = negative.sample(n=positive.shape[0])
        dataset = pd.concat([positive, negative])
        dataset.sample(frac=1).reset_index(drop=True)  # shuffle

    rawdata_file = './data/middle/middle_result.csv'
    os.system('mkdir ./data/middle')
    os.system('ls -l ./data')
    print('writing middle process data {}...'.format(rawdata_file))
    dataset.to_csv(rawdata_file, index=False)

    return dataset