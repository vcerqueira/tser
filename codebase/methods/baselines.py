import copy
from typing import Dict

import pandas as pd
from sklearn.multioutput import MultiOutputRegressor

from codebase.methods.tser import TSERModel


class Baseline:
    def __init__(self, model):
        self.model = copy.deepcopy(MultiOutputRegressor(model))
        self.columns = []

    def fit(self, train_dict: Dict):
        """
        :param train_dict: Training set. A dict with the collection of time series for training
        """
        pass

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)


class LocalModel(Baseline):

    def __init__(self, model, target_entity: str):
        super().__init__(model)

        self.target_entity = target_entity

    def fit(self, train_dict: Dict):
        train_dict_ = train_dict.copy()

        assert self.target_entity in train_dict_

        training = train_dict_[self.target_entity]

        assert isinstance(training, pd.DataFrame)

        y_cols = TSERModel.get_target_columns(training.columns)

        Y = training[y_cols]
        X = training.drop(y_cols, axis=1)

        self.columns = X.columns
        self.model.fit(X.values, Y)


class GlobalModel(Baseline):

    def __init__(self, model):
        super().__init__(model)

    def fit(self, train_dict: Dict):
        train_df = pd.concat(train_dict, axis=0, copy=True)

        y_cols = TSERModel.get_target_columns(train_df.columns)

        Y = train_df[y_cols]
        X = train_df.drop(y_cols, axis=1)

        self.columns = X.columns
        self.model.fit(X.values, Y)
