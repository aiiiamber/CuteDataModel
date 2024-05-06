# -*- coding: utf-8 -*-

from __future__ import division

import traceback
import math
import numpy as np
from numpy import sqrt as np_sqrt
from numpy.random import choice, random

import pandas as pd

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)

import seaborn as sns
import random
from collections import defaultdict
from copy import deepcopy

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn import metrics
from sklearn.metrics import r2_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from imblearn.under_sampling import RandomUnderSampler

import scipy.stats as ss
from scipy.special import comb
from scipy.stats import norm, t
from scipy.stats.distributions import chi2

import dowhy
from dowhy.causal_graph import CausalGraph
from dowhy import CausalModel
import networkx as nx


class dml_runner:

    def __init__(self, params):

        self.data_set = params['data_set']
        self.data = pd.read_csv(self.data_set)

        assert len(self.data) == len(self.data.dropna()), 'dataset contains nulls!'
        self.cleaner()

        # all passed assumed to be a list
        self.pre_Xs = params['pre_metrics']
        self.treatment = params['treatment']
        self.outcomes = params['outcomes']

        # must keep 传字典, key:outcome, value : [x of selection]
        self.must_keep_X = params['must_keep_X']
        self.X_df = self.data[self.pre_Xs]

        # pattern, key:outcome, value:x of selection
        self.X_selection_recorder = {}

    def cleaner(self):
        for item in self.data.columns:
            self.data[item] = self.data[item].astype('float')

    def remove_cols_with_no_info(self):
        # 剔除所有用户取值相等 或者 原最小值为缺失最大值为0 的异常字段
        desc = self.X_df.describe(include='all', percentiles=[.25, .5, .75, .90, .99])
        column_to_be_deleted = [_ for _ in desc.columns if (desc.loc["min", _] == desc.loc["max", _]) or (
                    desc.loc["min", _] == -1 and desc.loc["max", _] == 0)]

        return column_to_be_deleted

    def correl_remover(self):
        # 特征间相关性
        corr = self.X_df.corr()
        metric_list = []
        total_df_stage_2_column = list(self.X_df.columns)

        for x in list(corr.index):
            for y in list(corr.columns):
                if x != y and abs((corr.loc[x, y])) > 0.95 and (
                        total_df_stage_2_column.index(x) < total_df_stage_2_column.index(y)):
                    metric_list.append([x, y])

        remove_list = []

        for item in metric_list:
            # 无脑去除第一个
            remove_list.append(item[0])

        return remove_list

    def target_var_col(self):
        # 二分类特征选择专用
        # 按照与目标高相关的选择特征，强制留10个
        X_class = self.X_df
        y_class = self.data[self.treatment]

        sel_class = SelectKBest(score_func=f_classif, k=10)
        sel_class.fit(X_class, y_class)

        list_from_class = [self.pre_Xs[_] for _ in sel_class.get_support(True)]
        list_to_del = [i for i in self.pre_Xs if i not in list_from_class]

        return list_to_del

    def target_var_col2(self, outcome):
        # 回归特征选择专用
        # 按照与目标高相关的选择特征，强制留10个

        X_reg = self.X_df
        z = self.data[outcome]

        sel_reg = SelectKBest(score_func=f_regression, k=10)
        sel_reg.fit(X_reg, z)

        list_from_reg = [self.pre_Xs[_] for _ in sel_reg.get_support(True)]
        list_to_del = [i for i in self.pre_Xs if i not in list_from_reg]

        return list_to_del

    def undersampling(self, X, y):
        # 强制采样成1v1比例
        # 在训练集使用
        rus = RandomUnderSampler(random_state=random.randint(100000, 200000))
        X_resampled, y_resampled = rus.fit_resample(X, y)

        return X_resampled, y_resampled

    def train_test_splitter(self, selected_Xs, outcome):
        # 这里的x_class, 包含post指标
        x_columns = list(set([outcome] + selected_Xs))

        X_class_unob = self.data[x_columns]
        y_class_unob = self.data[self.treatment[0]]

        X_train_class_unob, X_test_class_unob, y_train_class_unob, y_test_class_unob = train_test_split(X_class_unob,
                                                                                                        y_class_unob,
                                                                                                        test_size=0.4,
                                                                                                        random_state=0)

        X_train_class_unob, y_train_class_unob = self.undersampling(X_train_class_unob, y_train_class_unob)
        X_test_class_unob, y_test_class_unob = self.undersampling(X_test_class_unob, y_test_class_unob)

        return X_train_class_unob, X_test_class_unob, y_train_class_unob, y_test_class_unob

    def binary_fitter(self, X_train_class_unob, X_test_class_unob, y_train_class_unob, y_test_class_unob, outcome):
        # 随机森林fitter，产出auc及混淆矩阵
        # X class传入之前必须去除post指标
        X_train_class_unob = X_train_class_unob[[i for i in X_train_class_unob.columns if i != outcome]]
        X_test_class_unob = X_test_class_unob[[i for i in X_test_class_unob.columns if i != outcome]]

        rf0_class_unob = RandomForestClassifier(n_estimators=200, oob_score=True,
                                                random_state=random.randint(100000, 200000))
        rf0_class_unob.fit(X_train_class_unob, y_train_class_unob)

        y_predclass_unob = rf0_class_unob.predict(X_test_class_unob)
        y_predprob_class_unob = rf0_class_unob.predict_proba(X_test_class_unob)[:, 1]

        auc_score = metrics.roc_auc_score(y_test_class_unob, y_predprob_class_unob)

        matrix_temp = confusion_matrix(list(y_test_class_unob), y_predclass_unob)
        # confusion_mat = ( matrix_temp[0,0]/ (matrix_temp[0,1] + matrix_temp[0,0])  , matrix_temp[1,1]/ (matrix_temp[1,1] + matrix_temp[1,0]) )

        return auc_score, matrix_temp

    def regressor(self, X_train_re, X_test_re, outcome):
        # 随机森林regressor
        # X class传入之前必须去除post指标

        y_train_re = X_train_re[outcome]
        y_test_re = X_test_re[outcome]

        X_train_re = X_train_re[[i for i in X_train_re.columns if i != outcome]]
        X_test_re = X_test_re[[i for i in X_test_re.columns if i != outcome]]

        rf_re = RandomForestRegressor(n_estimators=200, random_state=random.randint(100000, 200000))

        rf_re.fit(X_train_re, y_train_re)
        y_pred_re = rf_re.predict(X_test_re)

        r2_ = r2_score(y_test_re, y_pred_re)

        return r2_

    def control_all_cate(self, mydata, outcome):

        # data 里面，不能有其他的outcome; 必须要有treatment

        model_all_cate = CausalModel(data=mydata, treatment=self.treatment[0], outcome=outcome,
                                     common_causes=[i for i in mydata.columns if i not in [outcome, self.treatment[0]]])
        model_all_cate.view_model()

        identified_estimand_all_cate = model_all_cate.identify_effect(proceed_when_unidentifiable=True)
        print(identified_estimand_all_cate)

        econml_estimate_all_CATE = model_all_cate.estimate_effect(identified_estimand_all_cate,
                                                                  method_name="backdoor.econml.dml.LinearDML",
                                                                  control_value=0,
                                                                  treatment_value=1,
                                                                  confidence_intervals=True,
                                                                  target_units=1,
                                                                  method_params={
                                                                      'init_params': {'model_y': LinearRegression(),
                                                                                      'model_t': LinearRegression()},
                                                                      'fit_params': {}
                                                                  })

        absolute_effect = econml_estimate_all_CATE.value

        confidence_interval = '[ ' + str(np.mean(econml_estimate_all_CATE.effect_intervals[0])) + ' ,' + str(
            np.mean(econml_estimate_all_CATE.effect_intervals[1])) + ']'

        res_placebo_all_cate = model_all_cate.refute_estimate(identified_estimand_all_cate,
                                                              econml_estimate_all_CATE,
                                                              method_name="placebo_treatment_refuter",
                                                              placebo_type="permute",
                                                              num_simulations=100
                                                              # at least 100 is good, setting to 10 for speed
                                                              )

        res_subset_all_cate = model_all_cate.refute_estimate(identified_estimand_all_cate,
                                                             econml_estimate_all_CATE,
                                                             method_name="data_subset_refuter",
                                                             subset_fraction=0.8,
                                                             num_simulations=100
                                                             )

        res_random_all_cate = model_all_cate.refute_estimate(identified_estimand_all_cate,
                                                             econml_estimate_all_CATE,
                                                             method_name="random_common_cause"
                                                             )

        res_dummy_random_all_cate = model_all_cate.refute_estimate(identified_estimand_all_cate,
                                                                   econml_estimate_all_CATE,
                                                                   method_name="dummy_outcome_refuter"
                                                                   )[0]

        return absolute_effect, confidence_interval, res_placebo_all_cate, res_subset_all_cate, res_random_all_cate, res_dummy_random_all_cate

    def run(self):

        result_df = pd.DataFrame()
        # abnormal cols removal
        abnormal_cols = self.remove_cols_with_no_info()
        # highly correlated variables removal
        correl_cols = self.correl_remover()
        # compared vs treatment removal
        compared_with_treatment = self.target_var_col()

        # compared vs outcome removal
        for myoutcome in self.outcomes:
            compared_with_outcome = self.target_var_col2(myoutcome)

            # get final cols of selection
            out_X = list(set([i for i in self.pre_Xs if i not in [
                abnormal_cols + correl_cols + compared_with_treatment + compared_with_outcome]] + self.must_keep_X[
                                 myoutcome]))
            # train test split and undersampling
            X_train_class_unob, X_test_class_unob, y_train_class_unob, y_test_class_unob = self.train_test_splitter(
                out_X, myoutcome)
            # unobserved confounders evaluation for treatment
            rf_auc_score, rf_confusion_mat = self.binary_fitter(X_train_class_unob, X_test_class_unob,
                                                                y_train_class_unob, y_test_class_unob, myoutcome)
            # unobserved confounders evaluation for outcome
            rf_r2 = self.regressor(X_train_class_unob, X_test_class_unob, myoutcome)

            Xs = pd.concat([X_train_class_unob, X_test_class_unob], axis=0)
            ys = pd.concat([y_train_class_unob, y_test_class_unob], axis=0)
            mydata = pd.concat([Xs, ys], axis=1)

            absolute_effect, confidence_interval, res_placebo_all_cate, res_subset_all_cate, res_random_all_cate, res_dummy_random_all_cate = self.control_all_cate(
                mydata, myoutcome)

            x = np.average(self.data[self.data[self.treatment[0]] == float(1)][myoutcome])

            cate_percentage = float(absolute_effect) / (x - float(absolute_effect))

            result_df = result_df.append({'metric_name': myoutcome,
                                          'cate': absolute_effect,
                                          'cate_percentage': cate_percentage,
                                          'confidence_interval': confidence_interval,
                                          'X_used': out_X,
                                          'unobserved_confounders_treatment_AUC': rf_auc_score,
                                          'unobserved_confounders_treatment_confusion_mat': rf_confusion_mat,
                                          'unobserved_confounders_outcome_r2': rf_r2,
                                          'refutation_placebo': res_placebo_all_cate,
                                          'refutation_subset': res_subset_all_cate,
                                          'refutation_random_common_cause': res_random_all_cate,
                                          'refutation_dummy_outcome': res_dummy_random_all_cate
                                          }, ignore_index=True)
        return result_df








