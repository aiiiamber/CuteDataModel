# -*-coding:utf-8 -*-
# @Author: xiaolizhang

import yaml
import argparse

from model.builder import ModelBuilder
from utils.data_builder import preprocessing_data
from utils.config import Config


def print_data_log_info(conf, dataset):
    sample_nums = dataset.shape[0]
    label_value = dataset[conf.label_column].sum()
    if conf.task_type == 'binary':
        print('Label {} distribution >>'.format(conf.label_column))
        print('positive: {p_val:.2%}, negative: {n_val:.2%}'.format(p_val=label_value / sample_nums,
                                                                    n_val=1 - label_value / sample_nums))


def run():
    conf = Config()

    # build dataset
    dataset = preprocessing_data(conf=conf)
    print_data_log_info(conf, dataset)

    # build model
    model_builder = ModelBuilder()
    model = model_builder(conf=conf, dataset=dataset)

    # train model
    model.fit()

    # todo: model evaluation


if __name__ == '__main__':
    run()
