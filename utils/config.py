# -*-coding:utf-8 -*-
# @Author: xiaolizhang

import yaml
import argparse


class Config(object):

    def __init__(self):
        self._args = _parse_args()
        # load schema
        self.schema = self.load_yaml(self._args.schema_path)
        self._model_conf = self.load_yaml(self._args.model_path)

    @property
    def args(self):
        return _parse_args()

    @property
    def balanced(self):
        return self._args.balanced

    @property
    def task_type(self):
        return self._args.task_type

    @property
    def model_type(self):
        return self._args.model_type

    @property
    def model_conf(self):
        return self._model_conf

    def load_yaml(self, path):
        print("load yaml file {} ... ".format(path))
        return yaml.safe_load(open(path, 'r', encoding='utf8'))


def _parse_args():
    parser = argparse.ArgumentParser('Simplify Model')
    parser.add_argument('--data_input', type=str, default='data/test.csv')
    parser.add_argument('--schema_path', type=str, default='config/schema.yaml')
    parser.add_argument('--model_path', type=str, default='config/models.yaml')
    parser.add_argument('--task_type', type=str, default='binary', choices=['binary', 'regression'])
    parser.add_argument('--model_type', type=str, default='decision_tree', choices=['psm', 'decision_tree'])
    parser.add_argument('--balanced', type=bool, default=False,
                        help='If set, positive: negative = 1:1, otherwise keep it origin')
    args, _ = parser.parse_known_args()
    return args
