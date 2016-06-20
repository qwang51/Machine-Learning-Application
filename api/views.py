from api import app
import os
import pandas as pd
import numpy as np
import json
from flask import jsonify
from flask import render_template
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from flask import url_for


def get_abs_path():
    return os.path.abspath(os.path.dirname(__file__))


def get_data():
    f_name = os.path.join(get_abs_path(), 'data', 'breast-cancer-wisconsin.csv')
    # f_name = os.path.join(
    #     os.path.abspath(os.path.dirname(__file__)),
    #         'data',
    #         'breast-cancer-wisconsin.csv'
    # )

    columns = ['code', 'clump_thickness', 'size_uniformity', 'shape_uniformity', 'adhesion',
                'cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nuclei',
                'mitosis', 'class']
    df = pd.read_csv(f_name, sep=',', header=None, names=columns)
    df = df.convert_objects(convert_numeric=True)
    return df.dropna()


@app.route('/')
def index():
    df = get_data()
    X = df.ix[:, (df.columns != 'class') & (df.columns != 'code')].as_matrix()
    y = df.ix[:, df.columns == 'class'].as_matrix()
    # Scale
    scaler = preprocessing.StandardScaler().fit(X)
    scaled = scaler.transform(X)
    # PCA
    pcomp = decomposition.PCA(n_components=2)
    pcomp.fit(scaled)
    components = pcomp.transform(scaled)
    var = pcomp.explained_variance_ratio_.sum()  # View w/ Debug
    # KMeans
    model = KMeans(init='k-means++', n_clusters=2)
    model.fit(components)
    # Plot
    fig = plt.figure()
    plt.scatter(components[:, 0], components[:, 1], c=model.labels_)
    centers = plt.plot(
        [model.cluster_centers_[0, 0], model.cluster_centers_[1, 0]],
        [model.cluster_centers_[1, 0], model.cluster_centers_[1, 1]],
        'kx', c='Green'
    )
    # Increase size of center points
    plt.setp(centers, ms=11.0)
    plt.setp(centers, mew=1.8)
    # Plot axes adjustments
    axes = plt.gca()
    axes.set_xlim([-7.5, 3])
    axes.set_ylim([-2, 5])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Clustering of PCs ({:.2f}% Var. Explained'.format(
        var * 100
    ))
    # Save fig
    fig_path = os.path.join(get_abs_path(), 'static', 'tmp', 'cluster.png')
    fig.savefig(fig_path)
    return render_template('index.html', fig="hello")


@app.route('/d3')
def d3():
    df = get_data()
    X = df.ix[:, (df.columns != 'class') & (df.columns != 'code')].as_matrix()
    y = df.ix[:, df.columns == 'class'].as_matrix()
    # Scale
    scaler = preprocessing.StandardScaler().fit(X)
    scaled = scaler.transform(X)
    # PCA
    pcomp = decomposition.PCA(n_components=2)
    pcomp.fit(scaled)
    components = pcomp.transform(scaled)
    var = pcomp.explained_variance_ratio_.sum()  # View w/ Debug
    # KMeans
    model = KMeans(init='k-means++', n_clusters=2)
    model.fit(components)
    # Generate CSV
    cluster_data = pd.DataFrame(
        {'pc1': components[:,0],
         'pc2': components[:,1],
         'labels': model.labels_}
    )
    csv_path = os.path.join(get_abs_path(), 'static', 'tmp', 'kmeans.csv')
    cluster_data.to_csv(csv_path)
    return render_template('d3.html',
                           data_file=url_for('static',
                                               filename='tmp/kmeans.csv'))


def split_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test


def rfc(X_train, X_test, y_train):
    fit = RandomForestClassifier().fit(X_train, y_train)
    y_pred = fit.predict(X_test)
    y_score = fit.predict_proba(X_test)[:,1]
    return y_pred, y_score


@app.route('/prediction')
def prediction():
    df = get_data()
    X = df.ix[:, (df.columns != 'class') & (df.columns != 'code')].as_matrix()
    y = df.ix[:, df.columns == 'class'].as_matrix()
    y = np.where(y == 4, 1, 0)
    # Split training and testing set
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    # Random Forest
    y_pred, y_score = rfc(X_train, X_test, y_train)
    # ROC Plot
    fig = plt.figure()
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
    plt.plot(fpr, tpr, label='ROC curve (area = %f)'%metrics.roc_auc_score(y_test, y_score),
             lw=4, color="#0000ff", marker='s',markerfacecolor="red")
    plt.plot([0, 0], [1, 1])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    # Save fig
    fig_path = os.path.join(get_abs_path(), 'static', 'tmp', 'random_forest.png')
    fig.savefig(fig_path)
    return render_template('prediction.html', fig="ROC")


@app.route('/head')
def head():
    df = get_data().head()
    data = json.loads(df.to_json())
    return jsonify(data)


@app.route('/api/v1/prediction_confusion_matrix')
def confusion_matrix():
    df = get_data()
    X = df.ix[:, (df.columns != 'class') & (df.columns != 'code')].as_matrix()
    y = df.ix[:, df.columns == 'class'].as_matrix()
    # Split training and testing set
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    # Random Forest
    y_pred, y_score = rfc(X_train, X_test, y_train)
    cm = metrics.confusion_matrix(y_test, y_pred)
    d = {}
    d['fp'] = cm[0,1]
    d['tp'] = cm[1,1]
    d['fn'] = cm[0,1]
    d['tn'] = cm[0,0]
    result = {'random forest': d}
    return jsonify(result)