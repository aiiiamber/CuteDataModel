# -*-coding:utf-8 -*-
# @Author: xiaolizhang

import numpy as np
import pandas as pd


class ColumnArgs(object):

    def __init__(self, conf):
        self.schema = conf.schema
        # parse feature info
        self.column_names = list(self.schema.keys())
        self.label_column, self.index_column, self.vid_column, self.treatment_column = '', '', '', ''
        self.fc_columns, self.category_columns, self.numerical_columns = [], [], []
        self.parse_schema_basic_info()
        self.has_category = 0 if len(self.category_columns) == 0 else 1

    def parse_schema_basic_info(self):
        for key, value in self.schema.items():
            if value not in ['unused', 'label', 'index', 'exp', 'treatment']:
                self.fc_columns.append(key)
                if 'category' not in key:
                    self.numerical_columns.append(key)
                else:
                    self.category_columns.append(key)
            if value == 'label':
                self.label_column = key
            if value == 'index':
                self.index_column = key
            if value == 'exp':
                self.vid_column = key
            if value == 'treatment':
                self.treatment_column = key
