# -*-coding:utf-8 -*-
# @Author: xiaolizhang

import yaml
import argparse

from model.builder import ModelBuilder
from utils.feature_builder import ColumnArgs
from utils.data_builder import preprocessing_data
from utils.config import Config


def print_data_log_info(conf, fc, dataset):
    sample_nums = dataset.shape[0]
    label_value = dataset[fc.label_column].sum()
    if conf.task_type == 'binary':
        print('[Info] Label {} distribution >>'.format(fc.label_column))
        print('     positive label: {p_val:.2%}, \n'
              '     negative label: {n_val:.2%}'.format(p_val=label_value / sample_nums,
                                                        n_val=1 - label_value / sample_nums))


def run():
    conf = Config()

    # feature builder：generate feature column class info
    fc = ColumnArgs(conf=conf)

    # build dataset
    dataset = preprocessing_data(conf=conf, fc=fc)
    print_data_log_info(conf, fc, dataset)

    # build model
    model_builder = ModelBuilder()
    model = model_builder(conf=conf, fc=fc, dataset=dataset)

    # train model
    model.fit()

    # evaluate model
    model.evaluate()


if __name__ == '__main__':
    run()
