# -*-coding:utf-8 -*-
# @Author: xiaolizhang

from model.decision_tree import DecisionTree
from model.psm import PropensityScoreMatching


class ModelBuilder:

    def __int__(self):
        pass

    def __call__(self, conf, dataset):
        if conf.model_type == 'decision_tree':
            model = DecisionTree(conf, dataset)
            model.build()
        elif conf.model_type == 'psm':
            model = PropensityScoreMatching(conf, dataset)
            model.build()
