# coding =utf-8
# used with windows only - python version 3.6
#
#
#
# Pytest end to end machine learning model test framework

"""
Goal: To develop a test framework to test end to end machine learning.
Test cases: Ranges from testing train data variable type, checking return value of each function , validate range of certain column etc.
"""
import sys

import pandas.api.types as ptypes

import main
from pa_information import DataDescribe
from pa_model import Model


# _FILENAME = 'E:\\pa_work_sample_training_data.csv'

def test_get_train_and_test():
    X_train, X_test, y_train, y_test = Model._get_train_and_test ( DataDescribe.load_dataframe ( main._FILENAME ) )
    assert ptypes.is_float_dtype ( X_train [ 'Time spent' ] )


if __name__ == "__main__":
    test_get_train_and_test ()
    sys.exit ( 0 )
