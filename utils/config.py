# -*-coding:utf-8 -*-
# @Author: xiaolizhang

import yaml
import argparse


class Config(object):
    def __int__(self):
        self.args = _parse_args()
        # load schema
        self.schema = self.load_yaml(self.args.schema_path)
        # parse feature info
        self.column_names = list(self.schema.keys())
        self.label_column, self.index_column = '', ''
        self.fc_columns, self.category_columns, self.numerical_columns = [], [], []
        self.parse_schema_basic_info()
        self.has_category = 0 if len(self.category_columns) == 0 else 1

    def load_yaml(self, path):
        print("load yaml file {} ... ".format(path))
        return yaml.safe_load(open(path, 'r', encoding='utf8'))

    def parse_schema_basic_info(self):
        for key, value in self.schema.items():
            if value not in ['unused', 'label', 'index']:
                self.fc_columns.append(key)
                if 'category' not in key:
                    self.numerical_columns.append(key)
                else:
                    self.category_columns.append(key)
            if value == 'label':
                self.label_column = key
            if value == 'index':
                self.index_column = key


def _parse_args():
    parser = argparse.ArgumentParser('Simplify Model')
    parser.add_argument('--data_input', type=str, default='data/model/random_data.csv')
    parser.add_argument('--schema_path', type=str, default='config/schema.yaml')
    parser.add_argument('--task_type', type=str, default='binary', choices=['binary', 'regression'])
    parser.add_argument('--model_type', type=str, default='decision_tree', choices=['psm', 'decision_tree'])
    parser.add_argument('--balanced', type=bool, default=False,
                        help='If set, positive: negative = 1:1, otherwise keep it origin')
    args, _ = parser.parse_known_args()
    return args
