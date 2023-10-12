# -*-coding:utf-8 -*-
# @Author: xiaolizhang


class Core(object):
    def __init__(self):
        self._model = None

    def build(self, *args, **kwargs):
        return NotImplementedError

    def fit(self, *args, **kwargs):
        return NotImplementedError

    def predict(self, *args, **kwargs):
        return NotImplementedError


class BaseModel(Core):

    def __init__(self, conf, dataset):
        self._conf = conf
        self._dataset = dataset
        self._model = None
