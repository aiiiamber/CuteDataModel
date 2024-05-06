# -*-coding:utf-8 -*-
# @Author: xiaolizhang

import numpy as np
import lightgbm as lgb

from psmpy import PsmPy

from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def cal_auc(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_true[y_true > 0] = 1
    y_true[y_true < 0] = 0

    score = roc_auc_score(y_true, y_pred)
    return score


class OptimPsmPy(PsmPy):
    """
    Support multiple fit model for psm
    """

    def __init__(self, data, treatment, indx, exclude):
        super(OptimPsmPy, self).__init__(data, treatment, indx, exclude)
        self.category_columns = exclude

    def lightgbm_ps(self):
        # preprocessing data
        columns = self.data.columns.tolist()
        # columns.remove(self.exclude)

        idx, x, y = self.data[self.indx], self.data[columns], self.data[self.treatment]
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        train_data = lgb.Dataset(x_train, y_train, categorical_feature=self.category_columns)
        val_data = lgb.Dataset(x_test, y_test, categorical_feature=self.category_columns)
        # build model
        params = {
            'num_leaves': 50,
            'max_depth': 8,
            'min_data_in_leaf': 20,
            'min_child_samples': 20,
            'objective': 'binary',
            'learning_rate': 0.1,
            'boosting': 'gbdt',
            'feature_fraction': 0.8,
            'bagging_freq': 0,
            'bagging_fraction': 0.6,
            'bagging_seed': 23,
            'metric': 'auc',
            'lambda_l1': 0.2,
            'nthread': 4,
            'verbose': -1
        }
        clf = lgb.train(params=params,
                        train_set=train_data,
                        num_boost_round=1,
                        valid_sets=[train_data, val_data])

        # predict and merge
        self.predicted_data = x
        self.predicted_data['propensity_score'] = clf.predict(x)
        self.predicted_data['propensity_logit'] = self.predicted_data['propensity_score'].apply(
            lambda p: np.log(p / (1 - p)) if p < 0.9999 else np.log(p / (0.00001)))
        self.predicted_data[self.indx] = idx
        self.predicted_data[self.treatment] = y


class PropensityScoreMatching:

    def __init__(self, label_column, index_column, fc_columns, dataset, category_fc_columns=[], model_type='logit'):
        self.index_column = index_column
        self.label_column = label_column
        self.fc_column = fc_columns
        used_columns = fc_columns + [index_column, label_column]
        self.category_columns = category_fc_columns
        self.data = dataset[used_columns]
        self.model_type = model_type

    def build(self):
        self._model = OptimPsmPy(data=self.data,
                                 treatment=self.label_column,
                                 indx=self.index_column,
                                 exclude=[])

    def fit(self):
        # fit stage:
        if self.model_type == 'logit':
            self._model.logistic_ps(balance=False)  # todo: replace optim model
        elif self.model_type == 'lgb':
            self._model.lightgbm_ps()
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
        format_matched = format_matched[format_matched[self.index_column] != format_matched.matched_ID]
        return format_matched

    def evaluate(self):
        y_data = self._model.predicted_data[self.label_column]
        y_pred = self._model.predicted_data['propensity_score'].apply(lambda x: 1 if x > 0.5 else 0)
        print(classification_report(y_data, y_pred))
        print("metric AUC: {auc:.4f}".format(auc=cal_auc(y_data, y_pred)))
