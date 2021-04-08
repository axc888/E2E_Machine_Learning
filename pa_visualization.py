# coding =utf-8
# used with windows only - python version 3.6
#
#
#
# Visualization script -

"""
A simple dashboard web app

"""

import sys

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output

import proj.main as main
from proj.pa_information import DataDescribe


class Dashboard ():
    """
    This class give creates Dashboard for Software metrics.
    """
    external_stylesheets = [ dbc.themes.LUX ]
    df = DataDescribe.load_dataframe ( main._FILENAME )
    options = [ ]
    # option is used for getting a list of features in a dropdown
    # option consists features which are of numeric type
    for col in df.columns:
        if 'int64' == df [ col ].dtype or 'float64' == df [ col ].dtype:
            options.append ( { 'label': '{}'.format ( col ), 'value': col } )
    app = dash.Dash ( __name__, external_stylesheets = external_stylesheets )

    # Tabs styles
    tabs_styles = {
        'height': '51px'
    }
    # Normal style
    tab_style = {
        'borderBottom': '1px solid #d6d6d6',
        'padding': '2px',
        'fontWeight': 'bold'
    }
    # Tab style when selected
    tab_selected_style = {
        'borderTop': '1px solid #d6d6d6',
        'borderBottom': '1px solid #d6d6d6',
        'backgroundColor': 'black',
        'color': 'yellow',
        'padding': '10px'
    }
    # Basic html layout for the Dashboard.
    # It will contain labels for display as well as dynamic graphs & charts.
    app.layout = html.Div ( [
        dbc.Container ( [
            dbc.Row ( [
                dbc.Col ( html.H1 ( " Predictive Assessment",
                                    style = { 'background-color': '#3399ff', 'text-align': 'center' } ) )
            ] ),
            dbc.Row ( [
                dbc.Col ( html.H6 (
                    children = 'Please change the string',
                    style = { 'font-weight': 'bold', 'text-align': 'center' } ), className = "bold" )
            ] ),
            dbc.Row ( [
                dbc.Col ( dbc.Card (
                    html.H3 ( children = 'Visualization at a glance', className = "text-center text-light bg-dark" ),
                    body = True,
                    color = "dark" ) )
            ] ),
            html.Div ( id = 'dd-output-container' ),
            dcc.Tabs ( id = 'all-tabs-inline', value = 'tab-1', children = [
                dcc.Tab ( label = 'Tab one', value = 'tab-1', style = tab_style, selected_style = tab_selected_style ),
                dcc.Tab ( label = 'Tab two', value = 'tab-2', style = tab_style, selected_style = tab_selected_style ),
                dcc.Tab ( label = 'Tab three', value = 'tab-3', style = tab_style,
                          selected_style = tab_selected_style ),
                dcc.Tab ( label = 'Tab four', value = 'tab-4', style = tab_style, selected_style = tab_selected_style ),
            ], style = tabs_styles,
                       colors = {
                           "border": "yellow",
                           "primary": "red",
                           "background": "orange"
                       } ),

            html.Div ( id = 'tabs-content' )

            # dbc.Row([
            #     dbc.Col(dbc.Card(html.H6(children='Total no. of Developers')))
            # ])
        ] )
    ] )

    # app.callback is basically a decorator through which we tell Dash to call the function whenever the value of input
    # component such as dropdown or text box value changes & we want to update the children of the output component
    # on the page. Similar to html div components

    @app.callback (
        Output ( 'tabs-content', 'children' ),
        Input ( 'all-tabs-inline', 'value' ) )
    # Takes the dataframe and shows graphs in different tabs
    def render_content(tab):
        df = pd.read_csv ( 'E:\\pa_work_sample_training_data.csv' )
        options = [ ]
        for col in df.columns:
            if 'int64' == df [ col ].dtype or 'float64' == df [ col ].dtype:
                options.append ( { 'label': '{}'.format ( col ), 'value': col } )
        y_axis = {
            # 'title': 'Price',
            'showspikes': True,
            'spikedash': 'dot',
            'spikemode': 'across',
            'spikesnap': 'cursor',
        }

        x_axis = {
            # 'title': 'Time',
            'showspikes': True,
            'spikedash': 'dot',
            'spikemode': 'across',
            'spikesnap': 'cursor',
        }
        if tab == 'tab-1':
            return html.Div ( [
                html.H3 ( dcc.Graph (
                    id = '1',
                    figure = {
                        'data': [
                            { 'x': df [ 'tx_file_type' ].unique (), 'y': df [ 'tx_file_type' ].value_counts (),
                              'type': 'bar', 'name': 'tx_file_type' },
                        ],
                        'layout': {
                            'title': 'TASK FILE TYPE',
                            'height': 500,
                            'xaxis': x_axis,
                            'yaxis': y_axis,
                            'size': 14
                        }
                    }

                ) )
            ] )
        elif tab == 'tab-2':
            return html.Div ( [
                html.H3 ( dcc.Graph (
                    id = '2',
                    figure = {
                        'data': [
                            { 'x': df [ 'tx_difficulty' ].unique (), 'y': df [ 'tx_difficulty' ].value_counts (),
                              'type': 'bar', 'name': 'TX_DIFFICULTY' },
                        ],
                        'layout': {
                            'title': 'Task Difficulty',
                            'height': 500,
                            'xaxis': x_axis,
                            'yaxis': y_axis,
                            'size': 14
                        }
                    }

                ) )
            ] )
        elif tab == 'tab-3':
            return html.Div ( [
                html.H3 ( dcc.Graph (
                    id = '2',
                    figure = {
                        'data': [
                            { 'x': df [ 'Task Name' ].unique (), 'y': df [ 'Task Name' ].value_counts (), 'type': 'bar',
                              'name': 'TASK NAME' },
                        ],
                        'layout': {
                            'title': 'TASK NAME',
                            'height': 500,
                            'xaxis': x_axis,
                            'yaxis': y_axis,
                            'size': 14
                        }
                    }

                ) )
            ] )
        elif tab == 'tab-4':
            return html.Div ( [
                html.H3 ( dcc.Graph (
                    id = '2',
                    figure = {
                        'data': [
                            { 'x': df [ 'Remarks on Score' ].unique (), 'y': df [ 'Remarks on Score' ].value_counts (),
                              'type': 'bar', 'name': 'SF' },
                        ],
                        'layout': {
                            'title': 'Remarks on Score',
                            'height': 500,
                            'xaxis': x_axis,
                            'yaxis': y_axis,
                            'size': 14
                        }
                    }

                ) )
            ] )

    def update_output(value):
        return 'You have selected "{}"'.format ( value )

    app.run_server ( host = '127.0.0.1', port = 4580, debug = True )


Dashboard = Dashboard ()
sys.exit ( 0 )
