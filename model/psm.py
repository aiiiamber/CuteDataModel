# -*-coding:utf-8 -*-
# @Author: xiaolizhang

from model.base_model import BaseModel
from psmpy import PsmPy


class PropensityScoreMatching(BaseModel):

    def __init__(self, conf, dataset):
        super().__init__(conf=conf, dataset=dataset)

    def build(self):
        self._model = PsmPy(self._dataset,
                            treatment=self._conf.label_column,
                            indx=self._conf.index_column,
                            exclude=[])

    def fit(self):
        self._model.logistic_ps(balance=False)
        self._model.knn_matched(matcher='propensity_logit', replacement=True, caliper=None)

    def predict(self):
        df_matched = self._model.df_matched
        format_matched = df_matched.dropna()
        format_matched = format_matched[format_matched[self._conf.index_column] != format_matched.matched_ID]
        return format_matched
