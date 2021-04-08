# coding =utf-8
# used with windows only - python version 3.6
#
# This document contains proprietary information belonging to  Ltd
#
# Model api script -

"""
A simple web app to demonstrate  Predictive Assessment model via api

"""
import csv
import io
import sys
from io import StringIO

import joblib
import pandas as pd
from flask import Flask, make_response, request

import constant

app = Flask ( __name__ )


def transform(text_file_contents):
    return text_file_contents.replace ( "=", "," )


@app.route ( '/' )
def form():
    return """
        <html>
            <body>
                <h1><c>Assessment<c></h1>
                </br>
                </br>
                <p> Select a test csv file from Choose Button below. Please makes sure it does not have index and heading.Only features should be there.</p>
                <p> Click on Predict button and wait for few seconds & a dialog box will appear to save prediction.csv file. The output is avaiable in prediction column. </p>
                <form action="/transform" method="post" enctype="multipart/form-data">
                    <input type="file" name="data_file" class="btn btn-block"/>
                    </br>
                    </br>
                    <button type="submit" class="btn btn-primary btn-block btn-large">Predict</button>
                </form>
            </body>
        </html>
    """


@app.route ( '/transform', methods = [ "POST" ] )
def transform_view():
    f = request.files [ 'data_file' ]
    if not f:
        return "No file"

    stream = io.StringIO ( f.stream.read ().decode ( "UTF8" ), newline = None )
    csv_input = csv.reader ( stream )
    print ( csv_input )
    for row in csv_input:
        print ( row )

    stream.seek ( 0 )
    result = transform ( stream.read () )

    df = pd.read_csv ( StringIO ( result ) )

    clf = joblib.load ( constant._MODELPATH )
    df [ 'prediction' ] = clf.predict ( df )
    response = make_response ( df.to_csv () )
    response.headers [ "Content-Disposition" ] = "attachment; filename=prediction.csv"
    return response


if __name__ == "__main__":
    app.run ( debug = False, port = 8000 )
    sys.exit ( 0 )
