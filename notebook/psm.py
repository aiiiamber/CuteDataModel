# -*-coding:utf-8 -*-
# @Author: xiaolizhang

import pandas as pd
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

    def __init__(self, data, treatment, indx, exclude, train_data, control_data):
        super(OptimPsmPy, self).__init__(data, treatment, indx, exclude, train_data, control_data)
        self.category_columns = exclude
        self.model = None
        self.train_data = train_data
        self.control_data = control_data
        self.control_data[treatment] = 0

    def lightgbm_ps(self, num_boost_round=100):
        # preprocessing data
        columns = self.train_data.columns.tolist()
        columns.remove(self.treatment)
        columns.remove(self.indx)
        # print('feature columns >>\n{}'.format(',\n'.join(columns)))
        # columns.remove(self.exclude)

        print('split training data ...')
        idx, x, y = self.train_data[self.indx], self.train_data[columns], self.train_data[self.treatment]
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        train_data = lgb.Dataset(x_train, y_train, categorical_feature=self.category_columns)
        val_data = lgb.Dataset(x_test, y_test, categorical_feature=self.category_columns)

        # build model
        print('training model ...')
        params = {
            'num_leaves': 32,
            'max_depth': 8,
            'min_data_in_leaf': 20,
            'min_child_samples': 20,
            'objective': 'binary',
            'learning_rate': 0.1,
            'boosting': 'gbdt',
            'feature_fraction': 0.8,
            'bagging_freq': 0,
            'bagging_fraction': 0.8,
            'bagging_seed': 23,
            'metric': 'auc',
            'lambda_l1': 0.2,
            'nthread': 4,
            'verbose': -1
        }
        clf = lgb.train(params=params,
                        train_set=train_data,
                        num_boost_round=num_boost_round,
                        valid_sets=[train_data, val_data])
        self.model = clf

        # evaluate data
        self.evaluate_data = x
        self.evaluate_data['propensity_score'] = clf.predict(x)
        self.evaluate_data['propensity_logit'] = self.evaluate_data['propensity_score'].apply(
            lambda p: np.log(p / (1 - p)) if p < 0.9999 else np.log(p / (0.00001)))
        self.evaluate_data[self.indx] = idx
        self.evaluate_data[self.treatment] = y

        treatment_data = self.train_data[self.train_data[self.treatment] == 1]
        idx_treatment, x_treatment, y_treatment = treatment_data[self.indx], treatment_data[columns], treatment_data[self.treatment]
        idx_control, x_control, y_control = self.control_data[self.indx], self.control_data[columns], self.control_data[self.treatment]
        # predict and merge
        self.predicted_data = pd.concat([x_treatment, x_control], ignore_index=True)
        self.predicted_data['propensity_score'] = clf.predict(self.predicted_data )
        self.predicted_data['propensity_logit'] = self.predicted_data['propensity_score'].apply(
            lambda p: np.log(p / (1 - p)) if p < 0.9999 else np.log(p / (0.00001)))
        self.predicted_data[self.indx] = pd.concat([idx_treatment, idx_control], ignore_index=True)
        self.predicted_data[self.treatment] = pd.concat([y_treatment, y_control], ignore_index=True)


class PropensityScoreMatching:

    def __init__(self, label_column, index_column, fc_columns, dataset, control_data, model_config,
                 category_fc_columns=[]):
        self.index_column = index_column
        self.label_column = label_column
        self.fc_columns = fc_columns
        self.category_columns = category_fc_columns
        used_columns = fc_columns + [index_column, label_column]
        self.data = dataset[used_columns]
        self.control_data = control_data[used_columns]
        self.model_config = model_config
        self.model_type = model_config['model_type']

    def build(self):
        self._model = OptimPsmPy(data=self.data,
                                 treatment=self.label_column,
                                 indx=self.index_column,
                                 exclude=self.category_columns,
                                 train_data=self.data,
                                 control_data=self.control_data)

    def fit(self):
        # fit stage:
        if self.model_type == 'logit':
            print('Logistic Regression model is training ... ')
            self._model.logistic_ps(balance=False)
        elif self.model_type == 'lgb':
            print('LightGBM model is training ... ')
            self._model.lightgbm_ps(num_boost_round=self.model_config['num_boost_round'])
        # matching stage:
        print('KNN matching')
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
        return self._model.predicted_data, format_matched

    def predict_propensity_score(self, predicted_data, saved_columns=[]):
        model = self._model.model
        predicted_data['propensity_score'] = model.predict(predicted_data[self.fc_columns])
        predicted_data['binary_predicted_label'] = predicted_data['propensity_score'].apply(lambda x: 1 if x > 0.5
        else 0)
        default_saved_columns = [self.index_column, self.label_column, 'propensity_score', 'binary_predicted_label']
        return predicted_data[default_saved_columns + saved_columns]

    def evaluate(self):
        y_data = self._model.evaluate_data[self.label_column]
        y_pred = self._model.evaluate_data['propensity_score'].apply(lambda x: 1 if x > 0.5 else 0)
        print(classification_report(y_data, y_pred))
        print("metric AUC: {auc:.4f}".format(auc=cal_auc(y_data, y_pred)))
