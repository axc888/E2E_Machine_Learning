# coding =utf-8
# used with windows only - python version 3.6
#
#
#
# Data Information script -

import pandas as pd

from setup_logger import logging


class DataDescribe ():
    """
    This class give some basic information about the dataset.
    """

    def __init__(self, predictor=None, *args, **kwargs):
        predictor = 'avg_bce'
        self.predictorcol = predictor
        logging.info ( "Data Description object created" )

    def load_dataframe(self, filename):
        """
        Find missing values of given data
        :param filename: accepts csv file
        :return: Pandas Dataframe
        """
        df = pd.DataFrame ()
        try:
            # df = pd.read_csv('E:\\pa_work_sample_training_data.csv')
            df = pd.read_csv ( filename )
            logging.info ( f'Shape of loaded Dataframe: {df.shape}' )
            # print(df.head())
            logging.info ( "Dataframe loaded successfully" )
        except Exception as e:
            logging.error ( "Exception occurred", exc_info = True )
            logging.info ( "Dataframe loaded unsuccessfully" )
        return df

    def _get_missing_values(self, data):
        """
        Find missing values of given data
        :param data: checked its missing value
        :return: Pandas Series object
        """
        # Getting sum of missing values for each feature
        missing_values = data.isnull ().sum ()
        # Feature missing values are sorted from few to many
        missing_values.sort_values ( ascending = False, inplace = True )
        # Returning missing values
        return missing_values

    def _get_info(self, data):
        """
        print name of feature, their data type, number of missing values and ten samples of
        each feature
        :param data: dataset information will be gathered from
        :return: no return value
        """
        feature_dtypes = data.dtypes
        self.missing_values = self._get_missing_values ( data )

        print ( "=" * 200 )

        print ( "{:25} {:25} {:25} {:25}".format ( "Feature Name".upper (),
                                                   "Data Format".upper (),
                                                   "# of Missing Values".upper (),
                                                   "Samples".upper () ) )
        # Displays feature name, data type, missing value if any & 10 values from each feature
        for feature_name, dtype, missing_value in zip ( self.missing_values.index.values,
                                                        feature_dtypes [ self.missing_values.index.values ],
                                                        self.missing_values.values ):
            print ( "{:25} {:25} {:25} ".format ( feature_name, str ( dtype ), str ( missing_value ) ), end = "" )
            for value in data [ feature_name ].values [ :10 ]:
                print ( value, end = "," )
            print ()

        print ( "=" * 200 )

    def _get_unique_values(self, data):
        """
        print unique records of each feature
        :param data: dataset information will be gathered from
        :return: no return value
        """
        print ( "=" * 200 )
        print ( "{:10} {:25}".format ( "Feature Name".upper (),
                                       "Unique values".upper () ) )
        for feature_name in self.missing_values.index.values:
            print ( "{:25}".format ( feature_name ), end = "" )
            print ( data [ feature_name ].unique () [ :10 ], end = "," )
            print ()

    def _get_categorical_feature(self, data):
        """
        get list of categorical feature
        :param data: Pandas Dataframe
        :return: list
        """
        categorical_column = [ ]
        for col in data:
            if data [ col ].dtypes != "float64" and data [ col ].dtypes != "int64":
                categorical_column.append ( col )
        return categorical_column

    def _get_numeric_feature(self, data):
        """
        get list of numeric feature
        :param data: Pandas Dataframe
        :return: list
        """
        numeric_column = [ ]
        for col in data:
            if data [ col ].dtypes == "float64" and data [ col ].dtypes == "int64":
                numeric_column.append ( col )
        return numeric_column


DataDescribe = DataDescribe ()
