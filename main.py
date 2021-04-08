# coding =utf-8
# used with windows only - python version 3.6
#
#
#
# Main script to run project

import sys

import pandas as pd

import constant
from pa_feature_processing import FeatureProcessing
from pa_information import DataDescribe
from setup_logger import logging

# Read data from csv file
# Options
pd.set_option ( 'display.max_columns', 100 )
pd.set_option ( 'display.max_rows', 200 )


# Get basic information of dataframe
def df_description():
    df = DataDescribe.load_dataframe ( constant._FILENAME )
    logging.info ( DataDescribe._get_missing_values ( df ) )
    logging.info ( DataDescribe._get_info ( df ) )
    logging.info ( DataDescribe._get_unique_values ( df ) )


def visualization():
    # Dashboard()
    return 0


if __name__ == '__main__':
    data = DataDescribe.load_dataframe ( constant._FILENAME )
    model, X_train, X_test = FeatureProcessing.benchmark_model ( data, cross_validation = False,
                                                                 feature_selection = False )
    FeatureProcessing.model_explain ( X_train, X_test, model, data )
    sys.exit ( 0 )
