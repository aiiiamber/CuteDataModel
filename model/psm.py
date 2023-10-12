# -*-coding:utf-8 -*-
# @Author: xiaolizhang

from model.base_model import BaseModel
from psmpy import PsmPy


class OptimPsmPy(PsmPy):
    """
    Support multiple fit model for psm
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def lightgbm_ps(self):
        return None


class PropensityScoreMatching(BaseModel):

    def __init__(self, conf, fc, dataset):
        super().__init__(conf=conf, dataset=dataset)
        self.fc = fc

    def build(self):
        self._model = PsmPy(self._dataset,
                            treatment=self.fc.label_column,
                            indx=self.fc.index_column,
                            exclude=[])

    def fit(self):
        self._model.logistic_ps(balance=False)  # todo: replace model
        self._model.knn_matched(matcher='propensity_logit', replacement=True, caliper=None)

    def predict(self):
        # generate matching result
        df_matched = self._model.df_matched
        format_matched = df_matched.dropna()
        format_matched = format_matched[format_matched[self.fc.index_column] != format_matched.matched_ID]
        return format_matched
