# coding =utf-8
# used with windows only - python version 3.6
#
#
#
# Model generation script

import joblib
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import linear_model
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from setup_logger import logging


class Model:

    def __init__(self, *args, **kwargs):
        super ( Model, self ).__init__ ( *args, **kwargs )
        logging.info ( "Model object created" )

    # def _get_train_and_test(self,data):
    #     """
    #     Split data into train and test datasets
    #     :param data: Pandas Dataframe
    #     :return X_train: Pandas Dataframe
    #     :return X_test: Pandas Dataframe
    #     :return y_train: Pandas Series
    #     :return y_test: Pandas Series
    #     """
    #     X = FeatureProcessing.drop_columns(data, column_list = ['avg_bce'])
    #     y = data['avg_bce']
    #     # The number 42 is, in The Hitchhiker's Guide to the Galaxy by Douglas Adams, the "Answer to the Ultimate Question of Life, the Universe, and Everything", calculated by an enormous supercomputer named Deep Thought over a period of 7.5 million years.
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1,random_state=42)
    #     logger.info("Test train split done")
    #     return X_train, X_test, y_train, y_test

    def scale_quant_features(self, X_train, X_test):
        """
        Perform Standard Scalar
        :param X_train: Pandas Dataframe
        :param X_test: Pandas Dataframe
        :return X_train: Pandas Dataframe
        :return X_train: Pandas Dataframe
        """
        scaler = StandardScaler ()
        X_train = scaler.fit_transform ( X_train )
        X_test = scaler.transform ( X_test )
        return X_train, X_test

    def fit(self, model, X_train, y_train, parameters=None):
        """
        Perform Standard Scalar
        :param X_train: Pandas Dataframe
        :return X_train: Pandas Dataframe
        :return X_train: Pandas Dataframe
        """
        # model = model(**parameters)
        model.fit ( X_train, y_train )
        return model

    def predict(self, model, X_test):
        """
        perform prediction
        :param model: classifier
        :param X_test: Pandas Dataframe
        :return y_pred: Pandas Series
        """
        y_pred = model.predict ( X_test )
        return y_pred

    def pickle(self, model):
        """
        perform pickle operation
        :param model
        :return None
        """
        joblib.dump ( model, 'model.pkl' )

    def _mae(self, predict, target):
        """
        Mean absolute error
        :param predict
        :param target
        :return mae
        """
        return (abs ( predict - target )).mean ()

    def _mse(self, predict, target):
        """
        Mean square error
        :param predict
        :param target
        :return mse
        """
        return ((predict - target) ** 2).mean ()

    def _rmse(self, predict, target):
        """
        Root mean square error
        :param predict
        :param target
        :return rmse
        """
        return np.sqrt ( ((predict - target) ** 2).mean () )

    def _mape(self, predict, target):
        """
        Root mean square error
        :param predict
        :param target
        :return rmse
        """
        return (abs ( (target - predict) / target ).mean ()) * 100

    def _R2(self, predict, target):
        """
        Root mean square error
        :param predict
        :param target
        :return rsqaure: 1- SSR/SST
        """
        predict = predict
        target = target
        ss_residual = sum ( (target - predict) ** 2 )
        ss_total = sum ( (target - np.mean ( target )) ** 2 )
        r_squared = 1 - (float ( ss_residual )) / ss_total
        # logging.info ( f'R squared1: {r_squared}' )
        return r_squared
        # return 1 - (self._mae(predict,target) / self._mae( target.mean (), target))

    # R^2 explains the proportion of the variation in your dependent variable (Y) explained by your independent
    # variables (X) for a regression model.

    # While adjusted R^2 says the proportion of the variation in your dependent variable (Y) explained by more than 1
    # independent variables (X) for a  regression model.
    def R2_ADJ(self, predict, target, train):
        """
        Root mean square error
        :param predict
        :param target
        :param train
        :return adjusted rsqaure
        """
        r2 = self._R2 ( predict, target )
        # n = len(target)
        adj_r2 = (1 - (1 - r2) * ((train.shape [ 0 ] - 1) /
                                  (train.shape [ 0 ] - train.shape [ 1 ] - 1)))
        # return (1 - ((1 - r2) * ((n - 1) / (n - (k + 1)))))
        return adj_r2

    def cv_model_score(self, clf, train, labels):
        """
        Root mean square error
        :param clf: classifier
        :param train: Pandas Dataframe
        :param labels: Pandas Series
        :return score: float64
        """
        cv = KFold ( n_splits = 5, shuffle = True, random_state = 45 )
        r2 = make_scorer ( r2_score )
        r2_val_score = cross_val_score ( clf, train, labels, cv = cv, scoring = r2 )
        # adj_r2 = (1 - (1 - r2) * ((train.shape [ 0 ] - 1) /
        #                           (train.shape [ 0 ] - train.shape [ 1 ] - 1)))
        scores = [ r2_val_score.mean () ]

        # print ( clf.feature_importances_ )
        return scores

    def cv_test_models(self, train, labels):
        """
        Cross validation test different models
        :param train : Pandas Dataframe
        :param labels: Pandas Series
        :param results:  float64
        """
        results = { }
        clf = linear_model.LinearRegression ()
        results [ "Linear" ] = self.cv_model_score ( clf, train, labels )

        clf = linear_model.Ridge ()
        results [ "Ridge" ] = self.cv_model_score ( clf, train, labels )
        #
        clf = linear_model.BayesianRidge ()
        results [ "Bayesian Ridge" ] = self.cv_model_score ( clf, train, labels )

        clf = linear_model.HuberRegressor ()
        results [ "Hubber" ] = self.cv_model_score ( clf, train, labels )

        clf = linear_model.Lasso ( alpha = 1e-4 )
        results [ "Lasso" ] = self.cv_model_score ( clf, train, labels )

        clf = BaggingRegressor ()
        results [ "Bagging" ] = self.cv_model_score ( clf, train, labels )

        clf = RandomForestRegressor ()
        results [ "RandomForest" ] = self.cv_model_score ( clf, train, labels )
        # print(clf.feature_importances_)

        # clf = AdaBoostRegressor ()
        # results [ "AdaBoost" ] = self.cv_model_score( clf,train,labels )

        # clf = svm.SVR ()
        # results [ "SVM RBF" ] = self.cv_model_score( clf,train,labels )

        # clf = svm.SVR ( kernel = "linear" )
        # results [ "SVM Linear" ] = self.cv_model_score( clf,train,labels )

        results = pd.DataFrame.from_dict ( results, orient = 'index' )
        results.columns = [ "R Square Score" ]
        # results = results.sort(columns = ["R Square Score"], ascending = False)

        print ( results )
        return results


Model = Model ()
