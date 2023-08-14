# -*-coding:utf-8 -*-
# @Author: xiaolizhang

from model.base_model import BaseModel
from sklearn import tree
from sklearn.metrics import classification_report


class DecisionTree(BaseModel):

    def __int__(self, conf, dataset):
        super().__int__(conf=conf, dataset=dataset)
        params = self._conf.get('decision_tree', {})
        self.criterion = params.get("criterion", "entropy")
        self.max_depth = params.get("max_depth", 4)
        self.min_samples_split = params.get("min_samples_split", 1000)
        self.x_data = self._dataset[self._conf.fc_columns]
        self.y_data = self._dataset[[self._conf.label_column]]

    def build(self):
        self._model = tree.DecisionTreeClassifier(criterion=self.criterion,
                                                  max_depth=self.max_depth,
                                                  min_samples_split=self.min_samples_split)

    def fit(self):
        self._model.fit(self.x_data, self.y_data)
        self.print_evaluation_result()

    def predict(self, test_data):
        y_pred = self._model.predict(test_data)
        return y_pred

    def print_evaluation_result(self):
        fc = self._model.feature_importances_
        print("feature importance >> \n")
        for fn, fc in zip(self._conf.fc_columns, fc):
            if fc > 0:
                print('feature name: {}, feature importance: {}'.format(fn, fc))
        print("classification report >> \n")
        y_pred = self._model.predict(self.x_data)
        print(classification_report(self.y_data, y_pred, target_names=[0, 1]))