# -*-coding:utf-8 -*-
# @Author: xiaolizhang

from model.base_model import BaseModel

from psmpy import PsmPy
from sklearn.metrics import classification_report
from utils.metrics import cal_auc

import lightgbm as lgb


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
        self.data = self._dataset[[fc.label_column, fc.index_column] + fc.fc_columns]

    def build(self):
        self._model = PsmPy(self.data,
                            treatment=self.fc.label_column,
                            indx=self.fc.index_column,
                            exclude=self.fc.category_columns)

    def fit(self):
        # fit stage:
        self._model.logistic_ps(balance=False)  # todo: replace optim model
        # matching stage:
        self._model.knn_matched(matcher='propensity_logit', replacement=True, caliper=None)

    def predict(self):
        """ Matching ID
        model predicted result stored in `predicted_data`
        :return: pd.DataFrame
        """
        # generate matching result
        df_matched = self._model.df_matched
        format_matched = df_matched.dropna()
        format_matched = format_matched[format_matched[self.fc.index_column] != format_matched.matched_ID]
        return format_matched

    def evaluate(self):
        y_data = self._model.predicted_data[self.fc.label_column]
        y_pred = self._model.predicted_data['propensity_score'].apply(lambda x: 1 if x > 0.5 else 0)
        print(classification_report(y_data, y_pred))
        print("metric AUC: {auc:.4f}".format(auc=cal_auc(y_data, y_pred)))
