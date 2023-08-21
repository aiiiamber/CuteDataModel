# -*-coding:utf-8 -*-
# @Author: xiaolizhang

from model.decision_tree import DecisionTree
from model.psm import PropensityScoreMatching

class Core(object):
    def __int__(self):
        self._model = None

    def build(self, *args, **kwargs):
        return NotImplementedError

    def fit(self, *args, **kwargs):
        return NotImplementedError

    def predict(self, *args, **kwargs):
        return NotImplementedError


class BaseModel(Core):

    def __int__(self, conf, dataset):
        self._conf = conf
        self._dataset = dataset
        self._model = None


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