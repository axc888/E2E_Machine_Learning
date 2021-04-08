# coding =utf-8
# used with windows only - python version 3.6
#

#
# Feature Processing script -


import featuretools as ft
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from featuretools import variable_types as vtypes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import constant
from pa_information import DataDescribe
from pa_model import Model
from setup_logger import logging


class FeatureProcessing ( object ):
    """
    This class does the basic Feature Processing.
    """

    def __init__(self, *args, **kwargs):
        super ( FeatureProcessing, self ).__init__ ( *args, **kwargs )
        logging.info ( "FeatureProcessing object created" )

    # Drop features
    def drop_columns(self, data, column_list=None):
        """
        Drop features of given data
        :param data: Pandas datframe
        :param column_list: Python List,Default: None
        :return: Pandas Dataframe
        """
        for col in column_list:
            data = data.drop ( labels = [ col ], axis = 1 )
        return data

    def _label_encoder(self, data, column_list=None):
        """
        Perform label encode on given column list of a given data
        :param data: Pandas datframe
        :param column_list: Python List,Default: None
        :return: Pandas Dataframe
        """
        labelEncoder = LabelEncoder ()
        for col in column_list:
            data [ col ] = labelEncoder.fit_transform ( data [ col ].astype ( str ) )
        return data

    # Do one hot encoding
    def _get_dummies(self, data, column_list=None):
        """
        Perform one hot encoding on given column list of a given data
        :param data: Pandas datframe
        :param column_list: Python List,Default: None
        :return: Pandas Dataframe
        """
        if column_list is None:
            columns = data.columns.values
            non_dummies = None
        else:
            non_dummies = [ col for col in data.columns.values if col not in column_list ]
            columns = column_list
        dummies_data = [ pd.get_dummies ( data [ col ], prefix = col ) for col in columns ]
        if non_dummies is not None:
            for non_dummy in non_dummies:
                dummies_data.append ( data [ non_dummy ] )
        return pd.concat ( dummies_data, axis = 1 )

    # Create feature by multiplying two features
    def _get_multiply(self, data, first, second, new_column=None):
        """
        Perform mulplication on first & second column and generate new_column
        :param data: Pandas datframe
        :return: Pandas Dataframe
        """
        if new_column is not None:
            data [ new_column ] = data [ first ] * data [ second ]
        return data

    def _get_groupby_df(self, data, group_key, list_column, aggregate):
        """
        Perform groupby  on the column list given based on group key and aggregate function for eg.mean
        :param data: Pandas datframe
        :param group_key: Coulumn on which dataframe will be grouped
        :param list_column: List of column names to be grouped
        :param aggregate: Aggregate function used such as mean
        :return: Pandas Dataframe
        """
        for col in list_column:
            data [ aggregate + group_key + col ] = data.groupby ( [ col ] ) [ group_key ].transform ( aggregate )
        return data

    def _get_train_and_test(self, data):
        """
        Split data into train and test datasets
        :param data: Pandas Dataframe
        :return X_train: Pandas Dataframe
        :return X_test: Pandas Dataframe
        :return y_train: Pandas Series
        :return y_test: Pandas Series
        """
        X = FeatureProcessing.drop_columns ( data, column_list = [ 'avg_bce' ] )
        y = data [ 'avg_bce' ]
        # The number 42 is, in The Hitchhiker's Guide to the Galaxy by Douglas Adams, the "Answer to the Ultimate Question of Life, the Universe, and Everything", calculated by an enormous supercomputer named Deep Thought over a period of 7.5 million years.
        X_train, X_test, y_train, y_test = train_test_split ( X, y, test_size = 0.1, random_state = 42 )
        logging.info ( "Test train split done" )
        return X_train, X_test, y_train, y_test

    #
    def _get_correlated_columns(self, data):
        """
        Get list of correlated features to drop from Pandas Dataframe
        :param data: Pandas datframe
        :return: Python List
        """
        corr_mat = data.corr ( method = 'pearson' )
        fig, axes = plt.subplots ( figsize = (12, 9) )
        sns.heatmap ( corr_mat [ (corr_mat >= 0.9) | (corr_mat <= -0.9) ], vmax = 1, vmin = -1,
                      square = True )
        # Upper Traingle data
        upper_tri = corr_mat.where ( np.triu ( np.ones ( corr_mat.shape ), k = 1 ).astype ( np.bool ) )
        # print(f"upper_tri: {upper_tri}")
        to_drop = [ column for column in upper_tri.columns if any ( upper_tri [ column ] > 0.95 ) ]
        # print(to_drop)
        return to_drop

    def auto_feature_engineering(self, data, list_drop_columns, list_one_hot, list_label_encoding):
        """
        Generate features automatically
        :param data: Pandas datframe
        :param list_drop_columns: List to contain column names to be dropped
        :param list_one_hot: List to contain column names to do one-hot-encoding
        :param list_label_encoding: List to contain column names to do label encoding
        :return: Python List
        """

        # Drop columns
        logging.info ( f'Shape before drop: {data.shape}' )
        data = FeatureProcessing.drop_columns ( data,
                                                column_list = list_drop_columns )
        logging.info ( f'Shape after drop: {data.shape}' )

        # Get data in numerical format
        # One hot encoding
        data = FeatureProcessing._get_dummies ( data, column_list = list_one_hot )
        # if 'tx_file_type_0'  in df.columns:
        #     df = FeatureProcessing.drop_columns ( data, column_list = [ 'tx_file_type_0'] )
        logging.info ( f'Shape after One hot encoding: {data.shape}' )

        # Label encoding
        data = FeatureProcessing._label_encoder ( data, column_list = list_label_encoding )
        logging.info ( f'Shape after label encoding: {data.shape}' )

        # Define the varibles type. If not featuretool can also automatically take it.
        pa_variable_types = {
            # 'tx_name': vtypes.Ordinal,
            # 'Task Name': vtypes.Ordinal,
            'tx_difficulty': vtypes.Ordinal
        }
        _inter = pd.cut ( data [ 'ggg' ], bins = [ 0, 0.4, 0.7, 1 ], labels = [ 1, 2, 3 ] )
        data.insert ( 5, '_intervals', _inter )
        score_percentage_inter = pd.cut ( data [ 'score_percentage' ], bins = [ 40, 50, 60, 70, 80, 90 ],
                                          labels = [ 1, 2, 3, 4, 5 ] )
        data.insert ( 5, 'score_pe', score_percentage_inter )
        data [ '_intervals' ] = pd.to_numeric ( data [ '_intervals' ], errors = 'coerce' )
        data [ 'score_pevals' ] = pd.to_numeric ( data [ 'score_percentage_intervals' ], errors = 'coerce' )
        data._intervals = data._intervals.astype ( np.float ).astype ( "Int64" )
        data.score_percentage_intervals = data.score_percentage_intervals.astype ( np.float ).astype ( "Int64" )
        # data.score_percentage_intervals = data.score_percentage_intervals.astype ( 'int64' )
        data = FeatureProcessing._get_multiply ( data, 'hhhhhl', 'nuhhhh', 'mul1' )
        data.to_csv ( 'data.csv', sep = ',', encoding = 'utf-8' )
        # data [ [ '_intervals' ] ] = data [ [ '_intervals' ] ].astype ( int)
        # data['_intervals'] = data[ 'Time spent' ].transform (lambda x: pd.qcut(x, 4, labels=range(1,5), duplicates='drop'))
        # data [ '_Bin_low' ] = pd.IntervalIndex ( _intervals ).left
        # data [ '_Bin_high' ] = pd.IntervalIndex ( _intervals ).right

        # score_percentage_intervals = data.groupby ( 'Task Name' ) [ 'score_percentage' ].transform ( pd.qcut(data.rank(method='first'),4))
        # data [ 'score_percentage_Bin_low' ] = pd.IntervalIndex ( _intervals ).left
        # data [ 'score_percentage_Bin_high' ] = pd.IntervalIndex ( _intervals ).right
        data.fillna ( 0, inplace = True )
        # data = FeatureProcessing.drop_columns ( data,
        #                                         column_list = ['Task Name'] )

        # Create an entity set
        logging.info ( f'Creating entityset' )
        es = ft.EntitySet ( "pa" )
        # Create an entity. Here we are adding entity "pa_tx_info" to our entity "pa"
        logging.info ( f'Adding entity to entityset' )

        es.entity_from_dataframe ( entity_id = "pa_1",
                                   dataframe = data,
                                   index = "id_infra_instan",
                                   variable_types = pa_variable_types )
        # Here we will use tx_name--Developers & "Task Name" as index for group by

        es.normalize_entity ( base_entity_id = "pa_1",
                              new_entity_id = "Task_1",
                              index = "Task" )
        es.normalize_entity ( base_entity_id = "pa_1",
                              new_entity_id = "success_1",
                              index = "success" )

        es.normalize_entity ( base_entity_id = "pa_1",
                              new_entity_id = "_intervals_1",
                              index = "_intervals" )

        es.normalize_entity ( base_entity_id = "pa_1",
                              new_entity_id = "score_percentage_intervals_1",
                              index = "score_percentage_intervals" )

        es.normalize_entity ( base_entity_id = "pa_1",
                              new_entity_id = "Score_1",
                              index = "Score" )

        es.normalize_entity ( base_entity_id = "pa_1",
                              new_entity_id = "difficulty_1",
                              index = "difficulty" )

        # Find interesting values for categorical variables, to be used to generate “where” clauses
        es.add_interesting_values ()
        # Plot your
        # es.plot()
        # Get feature_matrix & feature
        logging.info ( f'Get feature matrix' )
        feature_matrix, features = ft.dfs ( entityset = es,
                                            target_entity = "pa_1",
                                            verbose = True,
                                            approximate = '36d' )
        feature_matrix.fillna ( 0, inplace = True )

        feature_matrix.to_csv ( 'auto_feature_engg.csv', sep = ',', encoding = 'utf-8' )
        logging.info ( f'Automatic feature engineering done.Shape : {feature_matrix.shape}' )
        col_to_drop = FeatureProcessing._get_correlated_columns ( feature_matrix )
        logging.info ( f'Correlating columns to drop: {col_to_drop}' )
        logging.info ( f'Shape before correlated feature drop: {feature_matrix.shape}' )
        # feature_matrix = FeatureProcessing.drop_columns ( feature_matrix, column_list = col_to_drop )

        logging.info ( f'Shape after correlated feature drop: {feature_matrix.shape}' )
        # Get X_train, X_test, y_train, y_test
        # X_train, X_test, y_train, y_test = Model._get_train_and_test(features)
        return feature_matrix

    def manual_feature_engineering(self, data, list_drop_columns, list_one_hot, list_label_encoding):
        """
         Generate features
         :param data: Pandas datframe
         :param list_drop_columns: List to contain column names to be dropped
         :param list_one_hot: List to contain column names to do one-hot-encoding
         :param list_label_encoding: List to contain column names to do label encoding
         :return: Python List
         """
        # Label encoding
        data = FeatureProcessing._label_encoder ( data, column_list = list_label_encoding )

        # Drop columns
        logging.info ( f'Shape before drop: {data.shape}' )
        data = FeatureProcessing.drop_columns ( data,
                                                column_list = list_drop_columns )
        data = FeatureProcessing.drop_columns ( data,
                                                column_list = [ 'id_infra_instan' ] )
        logging.info ( f'Shape after drop: {data.shape}' )

        # One hot encoding
        data = FeatureProcessing._get_dummies ( data, column_list = list_one_hot )

        logging.info ( f'Shape after One hot encoding: {data.shape}' )
        data.to_csv ( 'data1.csv', sep = ',', encoding = 'utf-8' )
        # Feature new columns
        data = FeatureProcessing._get_multiply ( data, 'totatness', 'corrrm', 'correpct' )
        # Groupby on basis of
        groupbyfeature = [ 'Tae', 'culty', 'Temarks' ]
        col_list = [ 'null', 'nl', 'nall',
                     'n_all', 'x_all' ]

        for groupbyfeaturecol in groupbyfeature:
            data = FeatureProcessing._get_groupby_df ( data, groupbyfeaturecol, col_list, 'mean' )
            data = FeatureProcessing._get_groupby_df ( data, groupbyfeaturecol, col_list, 'max' )
            data = FeatureProcessing._get_groupby_df ( data, groupbyfeaturecol, col_list, 'median' )
            data = FeatureProcessing._get_groupby_df ( data, groupbyfeaturecol, col_list, 'skew' )

        data.fillna ( 0, inplace = True )
        logging.info ( f'Shape after feature engineering: {data.shape}' )

        col_to_drop = FeatureProcessing._get_correlated_columns ( data )
        logging.info ( f'Column to drop drop: {col_to_drop}' )
        logging.info ( f'Shape before correlated feature drop: {data.shape}' )
        # data = FeatureProcessing.drop_columns ( data, column_list = col_to_drop )

        logging.info ( f'Shape after correlated feature drop: {data.shape}' )
        data.to_csv ( 'manual_feature_engg.csv', sep = ',', encoding = 'utf-8' )
        return data

    def benchmark_model(self, data, cross_validation=False, feature_selection=False):
        """
        Function to benchmark and check scores of different models
        :param data: Pandas datframe
        :return Pandas Dataframe
        :return Pandas Dataframe
        :return Pandas Dataframe
        """
        regressor = RandomForestRegressor ( n_estimators = 10000, random_state = 0, n_jobs = -1, max_depth = 100 )
        # regressor = LinearRegression()
        # regressor = KNeighborsRegressor(n_neighbors=2)

        list_drop_columns = [ 'te', 'massion', 'idsion', 'isession',
                              'jments' ]
        list_label_encoding = [ 'ticulty', 'success', 'Score' ]
        list_one_hot = [ 'type' ]
        logging.info ( f'Shape before feature engineering: {data.shape}' )
        df = FeatureProcessing.manual_feature_engineering ( data, list_drop_columns, list_one_hot, list_label_encoding )
        # df = FeatureProcessing.auto_feature_engineering (data, list_drop_columns, list_one_hot, list_label_encoding)
        logging.info ( f'Shape after feature engineering: {df.shape}' )
        # then do train & test split
        X_train, X_test, y_train, y_test = FeatureProcessing._get_train_and_test ( df )
        if feature_selection:
            list_columns = [ 'te_1.SKEW(_1.nl)' ]
            X_train, X_test = FeatureProcessing.feature_selection ( X_train, X_test, list_columns )
            logging.info ( f'Shape of X_train after feature selection: {X_train.shape}' )
        logging.info ( f'Shape of X_train without feature selection: {X_train.shape}' )
        # then standard scalar
        # Standard Scale
        logging.info ( f'Doing Standard Scaling' )
        Model.scale_quant_features ( X_train, X_test )

        if not cross_validation:
            model = Model.fit ( regressor, X_train, y_train )
            y_pred = Model.predict ( regressor, X_test )
            logging.info ( f'mae: {Model._mae ( y_pred, y_test )}' )
            logging.info ( f'mse: {Model._mse ( y_pred, y_test )}' )
            logging.info ( f'R square: {Model._R2 ( y_pred, y_test )}' )
            logging.info ( f'Adjusted R Square: {Model.R2_ADJ ( y_pred, y_test, X_train )}' )
            print ()
        else:
            pass
            model = Model.cv_test_models ( X_train, y_train )
        logging.info ( f'Creating pickle of model' )
        Model.pickle ( model )
        # FeatureProcessing.save_csv(X_train,filename = X_train)
        # FeatureProcessing.save_csv(X_test, filename = X_test)
        X_train.to_csv ( 'x_train.csv', sep = ',', encoding = 'utf-8' )
        X_test.to_csv ( 'x_test.csv', sep = ',', encoding = 'utf-8' )
        logging.info ( f'Creating pickle of model' )
        for feature in zip ( [ col for col in X_train.columns ], regressor.feature_importances_ ):
            print ( feature )
        return model, X_train, X_test

    def model_explain(self, X_train, X_test, clf, data):
        class_names = X_train.columns
        categorical_features = DataDescribe._get_categorical_feature ( data )
        X_train = X_train.to_numpy ()
        X_test = X_test.to_numpy ()
        logging.info ( f'Explaining model' )
        explainer = lime.lime_tabular.LimeTabularExplainer ( X_train, feature_names = class_names,
                                                             class_names = [ 'a' ],
                                                             categorical_features = categorical_features,
                                                             verbose = True,
                                                             mode = 'regression' )
        for i in range ( len ( X_test ) ):
            exp = explainer.explain_instance ( X_test [ i ], clf.predict, num_features = 10 )
            location = constant._EXPLAIN_MODELPATH + '\lime' + str ( i ) + '.html'
            logging.info ( f'Model Prediction explaination for {X_test} has been saved at {location}' )
            exp.save_to_file ( location )

    def feature_selection(self, X_train, X_test, list_columns):
        X_train = X_train [ list_columns ]
        X_test = X_test [ list_columns ]
        return X_train, X_test

    # def save_csv(self,data,filename):
    #     data.to_csv( filename+'.csv', sep = ',', encoding = 'utf-8' )
    #     logging.info ( f'Saving csv: {filename}.csv' )


FeatureProcessing = FeatureProcessing ()
