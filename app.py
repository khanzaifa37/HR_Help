import os, sys, shutil, time
import pickle
from flask import Flask, request, jsonify, render_template,send_from_directory
import pandas as pd
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import urllib.request
import json
from geopy.geocoders import Nominatim

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)



@app.route('/')
def root():
    return render_template('index.html')

@app.route('/images/<Paasbaan>')
def download_file(Paasbaan):
    return send_from_directory(app.config['images'], Paasbaan)

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/work.html')
def work():
    return render_template('work.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/result.html', methods = ['POST'])
def predict():

    dataset=pd.read_csv('data.csv')
    data=pd.read_csv('data.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
    data['timestamp'] = pd.to_datetime(data['timestamp'], format = '%d/%m/%Y %H:%M:%S')
    # DATE TIME STAMP FUNCTION
    column_1 = data.ix[:,0]

    db=pd.DataFrame({"year": column_1.dt.year,
              "month": column_1.dt.month,
              "day": column_1.dt.day,
              "hour": column_1.dt.hour,
              "dayofyear": column_1.dt.dayofyear,
              "week": column_1.dt.week,
              "weekofyear": column_1.dt.weekofyear,
              "dayofweek": column_1.dt.dayofweek,
              "weekday": column_1.dt.weekday,
              "quarter": column_1.dt.quarter,
             })

    dataset1=dataset.drop('timestamp',axis=1)
    data1=pd.concat([db,dataset1],axis=1)
    data1.dropna(inplace=True)    
    #fitting of data
    X=data1.iloc[:,[1,2,3,4,6,16,17]].values
    y=data1.iloc[:,[10,11,12,13,14,15]].values
    #splitting of data
    
    #from sklearn.cross_validation import train_test_split
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=50)
    
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X,y)

    rfc = RandomForestClassifier(n_estimators=100)
    
    knn.fit(X,y)

    if request.method == 'POST':

        address = request.form['Location']
        geolocator = Nominatim(user_agent="E-BEATS")
        location = geolocator.geocode(address)
        print(location.address)
        lat=[location.latitude]
        log=[location.longitude]
        latlong=pd.DataFrame({'latitude':lat,'longitude':log})
        print(latlong)

        DT= request.form['timestamp']
        latlong['timestamp']=DT
        data=latlong
        cols = data.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        data = data[cols]

        data['timestamp'] = pd.to_datetime(data['timestamp'].astype(str), errors='coerce')
        data['timestamp'] = pd.to_datetime(data['timestamp'], format = '%d/%m/%Y %H:%M:%S')
        column_1 = data.ix[:,0]
        DT=pd.DataFrame({"year": column_1.dt.year,
              "month": column_1.dt.month,
              "day": column_1.dt.day,
              "hour": column_1.dt.hour,
              "dayofyear": column_1.dt.dayofyear,
              "week": column_1.dt.week,
              "weekofyear": column_1.dt.weekofyear,
              "dayofweek": column_1.dt.dayofweek,
              "weekday": column_1.dt.weekday,
              "quarter": column_1.dt.quarter,
             })
        data=data.drop('timestamp',axis=1)
        final=pd.concat([DT,data],axis=1)
        X=final.iloc[:,[1,2,3,4,6,10,11]].values
        
        my_prediction = knn.predict(X)
        
        if my_prediction[0][0] == 1:
            my_prediction='Predicted crime : Act 379-Robbery'
            #print("ROBBERY")
        elif my_prediction[0][1] == 1:
            my_prediction='Predicted crime : Act 13-Gambling'
            #print("GAMBLAING")
        elif my_prediction[0][2] == 1:
            my_prediction='Predicted crime : Act 279-Accident'
            #print("ACCIDENT")
        elif my_prediction[0][3] == 1:
            my_prediction='Predicted crime : Act 323-Violence'
            #print("VOILENCE")
        elif my_prediction[0][4] == 1:
            my_prediction='Predicted crime : Act 302-Murder'
            #print("MURDER")
        elif my_prediction[0][5] == 1:
            my_prediction='Predicted crime : Act 363-kidnapping'
            #print("KIDNAPPING")
        else:
            my_prediction='Place is safe no crime expected at that timestamp.'
            #print("SAFE")
        
        proba = np.max(knn.predict_proba(X))*100
        probr = 'Probability of occurance : '+str(proba)


    return render_template('result.html', prediction = my_prediction,  prob = probr)


if __name__ == '__main__':
    app.run(debug = True)
