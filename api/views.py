from api import app
import os
import pandas as pd
import json
from flask import jsonify


def get_data():
    f_name = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
            'data',
            'breast-cancer-wisconsin.csv'
    )

    columns = ['code', 'clump_thickness', 'size_uniformity', 'shape_uniformity', 'adhesion',
                'cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nuclei',
                'mitosis', 'class']
    df = pd.read_csv(f_name, sep=',', header=None, names=columns)
    return df.dropna()

@app.route('/')
def index():
    return "Hello, I'm an API!"



@app.route('/head')
def head():
    df = get_data().head()
    data = json.loads(df.to_json())
    return jsonify(data)
