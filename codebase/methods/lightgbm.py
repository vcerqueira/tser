import lightgbm as lgbm

from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit
from sklearn.base import BaseEstimator, RegressorMixin

PARAMETER_SET = \
    dict(num_leaves=[5, 10, 15, 30],
         max_depth=[-1, 3, 5, 10, 15],
         lambda_l1=[0, 0.1, 1, 100],
         lambda_l2=[0, 0.1, 1, 100],
         learning_rate=[0.05, 0.1, 0.2],
         min_child_samples=[15, 30, 50, 100],
         n_jobs=[1],
         linear_tree=[True, False],
         boosting_type=['gbdt'])

BEST_PARAMS = {'nn5_daily_without_missing': {'boosting_type': 'gbdt',
                                             'lambda_l1': 0.1,
                                             'lambda_l2': 0.1,
                                             'learning_rate': 0.1,
                                             'linear_tree': True,
                                             'max_depth': -1,
                                             'min_child_samples': 30,
                                             'n_jobs': 1,
                                             'num_boost_round': 100,
                                             'num_leaves': 30},
               'solar-energy': {'boosting_type': 'gbdt',
                                'lambda_l1': 0,
                                'lambda_l2': 1,
                                'learning_rate': 0.2,
                                'linear_tree': True,
                                'max_depth': 5,
                                'min_child_samples': 100,
                                'n_jobs': 1,
                                'num_boost_round': 100,
                                'num_leaves': 30},
               'taxi_30min': {'boosting_type': 'gbdt',
                              'lambda_l1': 100,
                              'lambda_l2': 100,
                              'learning_rate': 0.1,
                              'linear_tree': True,
                              'max_depth': -1,
                              'min_child_samples': 15,
                              'n_jobs': 1,
                              'num_boost_round': 100,
                              'num_leaves': 30},
               'traffic_nips': {'boosting_type': 'gbdt',
                                'lambda_l1': 0,
                                'lambda_l2': 0,
                                'learning_rate': 0.2,
                                'linear_tree': True,
                                'max_depth': 10,
                                'min_child_samples': 50,
                                'n_jobs': 1,
                                'num_boost_round': 100,
                                'num_leaves': 30},
               'electricity_nips': {'boosting_type': 'gbdt',
                                    'lambda_l1': 0,
                                    'lambda_l2': 0,
                                    'learning_rate': 0.2,
                                    'linear_tree': True,
                                    'max_depth': 10,
                                    'min_child_samples': 50,
                                    'n_jobs': 1,
                                    'num_boost_round': 100,
                                    'num_leaves': 30},
               'rideshare_without_missing': {'boosting_type': 'gbdt',
                                             'lambda_l1': 1,
                                             'lambda_l2': 100,
                                             'learning_rate': 0.2,
                                             'linear_tree': True,
                                             'max_depth': -1,
                                             'min_child_samples': 15,
                                             'n_jobs': 1,
                                             'num_boost_round': 50,
                                             'num_leaves': 30},
               'm4_hourly': {'boosting_type': 'gbdt',
                             'lambda_l1': 1,
                             'lambda_l2': 1,
                             'learning_rate': 0.2,
                             'linear_tree': True,
                             'max_depth': 5,
                             'min_child_samples': 30,
                             'n_jobs': 1,
                             'num_boost_round': 50,
                             'num_leaves': 30},
               'm4_weekly': {'boosting_type': 'gbdt',
                             'lambda_l1': 0.1,
                             'lambda_l2': 100,
                             'learning_rate': 0.2,
                             'linear_tree': True,
                             'max_depth': -1,
                             'min_child_samples': 50,
                             'n_jobs': 1,
                             'num_boost_round': 100,
                             'num_leaves': 10},
               }


class LightGBMOptim(BaseEstimator, RegressorMixin):

    def __init__(self, iters: int = 50, params=None):
        self.model = None
        self.iters = iters
        self.estimator = lgbm.LGBMRegressor(n_jobs=1)
        self.params = params
        self.parameters = \
            dict(num_leaves=[5, 10, 15, 30],
                 max_depth=[-1, 3, 5, 10],
                 lambda_l1=[0, 0.1, 1, 100],
                 lambda_l2=[0, 0.1, 1, 100],
                 learning_rate=[0.05, 0.1, 0.2],
                 min_child_samples=[15, 30, 50, 100],
                 n_jobs=[1],
                 linear_tree=[True, False],
                 boosting_type=['gbdt'],
                 num_boost_round=[25, 50, 100])

    def fit(self, X, y=None):
        if self.params is None:
            self.model = RandomizedSearchCV(estimator=self.estimator,
                                            param_distributions=self.parameters,
                                            scoring='neg_mean_squared_error',
                                            n_iter=self.iters,
                                            n_jobs=1,
                                            refit=True,
                                            verbose=0,
                                            cv=ShuffleSplit(n_splits=1, test_size=0.3),
                                            random_state=123)

            self.model.fit(X, y)
        else:
            self.model = lgbm.LGBMRegressor(**self.params, verbose=-1)
            self.model.fit(X, y)

    def predict(self, X):
        y_hat = self.model.predict(X)

        return y_hat

    @staticmethod
    def optimize_params(X, y):

        mod = LightGBMOptim(iters=200)

        mod.fit(X, y)

        return mod.model.best_params_
