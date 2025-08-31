from flask import Flask, render_template, request,render_template_string
import requests

import mysql.connector
import json
import textwrap
import urllib.request
from urllib.request import HTTPError

import cv2
from pyzbar import pyzbar

import speech_recognition as sr
from flask import redirect,url_for
import pyaudio
import time

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

app = Flask(__name__,template_folder='template')

# OpenWeatherMap API key
API_KEY = "65f7db79b97b88af39774a60c8263588"
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# INTIAL CODE FOR CROP RECOMMENDATION
# Load the dataset
data = pd.read_csv("Soil_dataset.csv")

# print("Dataset Columns:", data.columns.tolist())  # Debug line

def simple_recommender(crop_info):
    """
    Simple recommendation system that finds the most similar conditions in the dataset
    """

    input_df = pd.DataFrame([{
        'N': float(crop_info[0]),
        'P': float(crop_info[1]),
        'K': float(crop_info[2]),
        'temperature': float(crop_info[3]),
        'humidity': float(crop_info[4]),
        'ph': float(crop_info[5]),
        'rainfall': float(crop_info[6]),
        'wind_speed': float(crop_info[7])
    }])
    # Debug prints
    # print("Input DF:\n", input_df)
    # print("Available Columns:", data.columns.tolist())

    # Columns to compare
    compare_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'wind_speed']

    # Validate columns
    missing = [col for col in compare_cols if col not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    # Standardize dataset and input
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[compare_cols])
    input_scaled = scaler.transform(input_df[compare_cols])

    # Compute Euclidean distances
    distances = np.linalg.norm(data_scaled - input_scaled, axis=1)

    # Find top 5 closest entries
    top_indices = distances.argsort()[:5]
    top_crops = data.iloc[top_indices]['label'].value_counts().index.tolist()

    # Format and return recommendations

    result = {
        'top_recommendation': top_crops[0],
        'all_recommendations': [
            {'crop': crop, 'probability': f"{(len(top_crops) - i) / len(top_crops):.0%}"}
            for i, crop in enumerate(top_crops)
        ]
    }
    top_recom = result["top_recommendation"]
    probability = result["all_recommendations"][0]['probability']

    # print(top_recom)
    # print(all_recom)

    N = float(crop_info[0])
    P = float(crop_info[1])
    K = float(crop_info[2])
    temperature = float(crop_info[3])
    humidity = float(crop_info[4])
    ph = float(crop_info[5])

    rainfall = float(crop_info[6])
    wind_speed = float(crop_info[7])

    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="lsk12312",
        database="agriguide"
    )
    mycursor = db.cursor()

    query = (
            "INSERT INTO CROP_RECOMMENDATION(NITROGEN, PHOSPHOROUS, POTASSIUM, PH_VALUE, TEMPERATURE, HUMIDITY, RAINFALL, WIND_SPEED,CROP_NAME, PROBABILITY )VALUES('" + str(
        N) + "','" + str(P) + "','" + str(K) + "','" + str(
        ph) + "','" + str(temperature) + "','" + str(humidity) + "','" + str(rainfall) + "','" + str(
        wind_speed) + "','"+ str(top_recom) +"','"+str(probability)+"')")

    mycursor.execute(query)
    db.commit()

    print(result)

    return result


# FUNCTIONS CALLS
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('login.html')

@app.route('/weather',methods=['GET', 'POST'])
def weather():
    weather_data = None
    if request.method == 'POST':
        city = request.form.get('city')
        if city:
            params = {'q': city, 'appid': API_KEY, 'units': 'metric'}
            response = requests.get(BASE_URL, params=params)
            if response.status_code == 200:
                weather_data = response.json()
                print(weather_data)
            else:
                weather_data = {'error': 'City not found'}
                alert = """<html>
                            <head>
                            <script>
                            alert("CITY NOT FOUND...")
                            </script>
                            </head>
                            <body>
                            </body>
                            </html>
                            """
                return render_template_string(alert)

    return render_template('weather.html', weather_data=weather_data)

@app.route('/crop',methods=['GET', 'POST'])
def crop_page():
    return render_template('crop.html')

@app.route('/recommend',methods=['POST'])
def crop_recommend():

    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="lsk12312",
        database="agriguide"
    )
    mycursor = db.cursor()

    crop_info = [] #LIST
    if request.form['click']=='btn_click':

        N = float(request.form['nitrogen'])
        P = float(request.form['phosphorus'])
        K = float(request.form['potassium'])
        ph = float(request.form['ph'])

        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        rainfall = float(request.form['rainfall'])
        wind_speed = float(request.form['wind_speed'])

        weather_condition = str(request.form['weather_condition'])
        weather_sustainability = str(request.form['weather_sustainability'])

        # print("asdasd")
        query = (
                "INSERT INTO CROP_INFO(NITROGEN, PHOSPHOROUS, POTASSIUM, PH_VALUE, TEMPERATURE, HUMIDITY, RAINFALL, WIND_SPEED, WEATHER_CONDITION, WEATHER_SUSTAINABILITY )VALUES('" + str(N) + "','" + str(P) + "','" + str(K) + "','" + str(
            ph) + "','" + str(temperature) + "','" + str(humidity) + "','" + str(rainfall) + "','" + str(wind_speed) + "','"+str(weather_condition)+"','"+str(weather_sustainability)+"')")

        mycursor.execute(query)
        db.commit()

        crop_info=[N, P, K, temperature, humidity, ph, rainfall, wind_speed, weather_condition, weather_sustainability]

        # print(crop_info)

        recommendations = simple_recommender(crop_info)

    else:
        alert = """<html>
                    <head>
                    <script>
                   alert("Error in the Insertion of the Database")
                   </script>
                   </head>
                   <body>
                   </body>
                   <html>
                    """
        return render_template_string(alert)

    return render_template('recommend.html',top_recommendation=recommendations['top_recommendation'],
                           recommendations=recommendations['all_recommendations'],crop_info = crop_info)



    #  CREATE TABLE CROP_RECOMMENDATION(ID INT UNIQUE AUTO_INCREMENT,NITROGEN FLOAT,PHOSPHROUS FLOAT, POTASSIUM FLOAT, PH_VALUE FLOAT,TEMPERATURE FLOAT, HUMIDITY FLOAT, RAINFALL FLOAT, WIND_SPEED FLOAT, CROP_NAME VARCHAR(100), PROBABILITY VARCHAR(20));

    # return render_template('recommend.html',crop_info=crop_info)
# CREATE TABLE CROP_INFO(ID INT UNIQUE AUTO_INCREMENT,NITROGEN FLOAT,PHOSPHROUS FLOAT, POTASSIUM FLOAT, PH_VALUE FLOAT,TEMPERATURE FLOAT, HUMIDITY FLOAT, RAINFALL FLOAT, WIND_SPEED FLOAT,WEATHER_CONDITION VARCHAR(50), WEATHER_SUSTAINABILITY VARCHAR(50))


@app.route('/model',methods=['GET', 'POST'])
def model():

    return render_template('model.html')


# LOGIN BUTTON
@app.route('/login_button',methods=['POST'])
def login_button():
    global username,password
    global role
    global u,p
    u=""
    p=""
    # localhost_addr="https://localhost:5000"

    db=mysql.connector.connect(
        host="localhost",
        user='root',
        password='lsk12312',
        database='agriguide'
    )
    mycursor=db.cursor()

    username=request.form['username']
    password=request.form['password']
    role=request.form['role']

    print(username)
    print(password)
    print(role)

    if request.form['click']=='btn_click':

        time.sleep(3)
        select_query=f"SELECT * FROM USERS WHERE USERNAME='{username}'"
        mycursor.execute(select_query)
        existing_record=mycursor.fetchone()

        if(existing_record):
            try:
                username_query = f"SELECT USERNAME FROM USERS WHERE USERNAME='{username}'"
                mycursor.execute(username_query)  # type NONE
                username_record = mycursor.fetchone()
                # print(username_record)                   # username_record ---> tuple

                # If password is wrong Type Error Occurs

                u = "".join(username_record)  # tuple to str conversion
                # print(u)

                password_query=f"SELECT PASSWORD FROM USERS WHERE USERNAME='{username}' AND PASSWORD= '{password}' "
                # password_query = f"SELECT PASSWORD FROM USERS WHERE PASSWORD='{password}'"
                mycursor.execute(password_query)
                password_record = mycursor.fetchone()
                # print(password_record)              # password_record ---> tuple
                p = "".join(password_record)  # tuple to str conversion
                # print(p)

                if (u == username and password == p):
                    print("Correct user and pass")

                    #localhost_address=localhost_addr

                return render_template('index.html')    #Localhost address of isbn page


            except TypeError as e:
                incorrect_pass="""
                <html>
                <head>
                <script>
                alert("Incorrect Password.... ")
                </script>
                </head>
                <body>
                </body>
                </html>
                """
                # print("tada")
                return render_template_string(incorrect_pass)


        else:
            print("Please Register...")
            register_alert = """
                                <html>
                                <head>
                                <script>
                                alert("PLEASE REGISTER TO ACCESS...")
                                </script>
                                </head>
                                <body>
                                </body>
                                </html>
                                """

            return render_template_string(register_alert)


        # query="INSERT INTO USER(username,password,role) VALUES('"+username+"','"+password+"','"+role+"')"
    return render_template('login.html')

# SIGN UP LINK
@app.route('/signup')
def signup_link():
    print("Sign up link")
    return render_template('signup.html')

# Login Page⬆️
# Sign Up Page ⬇️

# SIGNUP BUTTON
@app.route('/signup_button', methods=['POST'])
def signup():
    global name,role,email
    global username,password

    print("Check point 2")

    db=mysql.connector.connect(
        host="localhost",
        user='root',
        password='lsk12312',
        database='agriguide'
    )
    mycursor=db.cursor()

    name=request.form['name']
    username=request.form['username']
    password=request.form['password']
    role=request.form['role']
    email=request.form['email']

    print(name)
    print(username)
    print(password)
    print(role)
    print(email)

    if request.form['click']=='btn_click':
        print("asdasd")
        query="INSERT INTO USERS(name,username,password,role,email) VALUES('"+name+"','"+username+"','"+password+"','"+role+"','"+email+"')"
        mycursor.execute(query)
        db.commit()
    return render_template('signup.html')

# LOGIN LINK
@app.route('/login')
def login_link():
    print("Login Back")
    return render_template("login.html")


if __name__ == '__main__':
    app.run(debug=True)

# return redirect(url_for('index'))
# @app.route('/api/recommend', methods=['POST'])
# def api_recommend():
#     try:
#         data = request.get_json()
#         input_data = {
#             'N': float(data['N']),
#             'P': float(data['P']),
#             'K': float(data['K']),
#             'temperature': float(data['temperature']),
#             'humidity': float(data['humidity']),
#             'ph': float(data['ph']),
#             'rainfall': float(data['rainfall']),
#             'wind_speed': float(data['wind_speed']),
#             'weather_condition': data['weather_condition'],
#             'weather_suitability': data['weather_sustainability']
#         }
#
#         recommendations = simple_recommender(input_data)
#         return jsonify(recommendations)
#
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400


# except Exception as e:
#     print(str(e))
#     return render_template('crop.html', error=str(e))
# N = 0
# P = 0
# K = 0
# temperature = 0
# ph = 0
# humidity = 0
# rainfall = 0
# wind_speed = 0
# weather_condition =0
# weather_sustainability =0
# input_data={}

# N = request.form['nitrogen']
# P = request.form['phosphorus']
# K = request.form['potassium']
# temperature = request.form['temperature']
# humidity = request.form['humidity']
# ph = request.form['humidity']
# rainfall = request.form['rainfall']
# wind_speed = request.form['wind_speed']
# weather_condition = request.form['weather_condition']
# weather_sustainability = request.form['weather_sustainability']

# print(type(N))
# print(type(P))
# print(type(K))
# print(type(temperature))
# print(type(humidity))
# print(type(ph))
# print(type(rainfall))
# print(type(wind_speed))
# print(type(weather_type))
# print(type(weather_condition))
# print(type(weather_sustainability))

# Get form data

    # # Check if all columns exist
    # missing = [col for col in compare_cols if col not in data.columns]
    # if missing:
    #     raise ValueError(f"Missing columns in dataset: {missing}")
    #
    # # Standardize the data for comparison
    # scaler = StandardScaler()
    # data_scaled = scaler.fit_transform(data[compare_cols])
    # input_scaled = scaler.transform(input_df[compare_cols])
    #
    # # Calculate Euclidean distances
    # distances = np.linalg.norm(data_scaled - input_scaled, axis=1)
    #
    # # Get top 5 most similar entries
    # top_indices = distances.argsort()[:5]
    # top_crops = data.iloc[top_indices]['label'].value_counts().index.tolist()
    #
    # # Format output
    # result = {
    #     'top_recommendation': top_crops[0],
    #     'all_recommendations': [
    #         {'crop': crop, 'probability': f"{(len(top_crops) - i) / len(top_crops):.0%}"}
    #         for i, crop in enumerate(top_crops)
    #     ]
    # }
    # return result


# sadpasd
#  # print(crop_info)
#     print("Input DF: ",input_df)
#
#     available_columns = data.columns.tolist()
#
#     print("Available Columns:",available_columns)
#     # print(type(input_df))   <class 'pandas.core.frame.DataFrame'>
#
#
#     # weather_condition = crop_info[8]
#     # weather_sustainability = crop_info[9]
#
#     # Columns expected
#     compare_cols = ['N','P','K','temperature','humidity','ph','rainfall','wind_speed']
#
#     # print(type(compare_cols)) # LIST
#     # compare_cols = crop_info
#     # print(compare_cols)
#
#     # Check if all columns exist
#     missing = [col for col in compare_cols if col not in data.columns]
#     if missing:
#         raise ValueError(f"Missing columns in dataset: {missing}")
#
#     # Standardize the data for comparison
#     scaler = StandardScaler()
#     data_scaled = scaler.fit_transform(data[compare_cols])
#
#     print("Data_scaled",data_scaled)
#     # print(type(scaler)) <class 'sklearn.preprocessing._data.StandardScaler'>
#     # print(type(data_scaled)) <class 'numpy.ndarray'>
#
#     input_scaled = scaler.transform(compare_cols)
#
#     # Calculate Euclidean distances
#     distances = np.linalg.norm(data_scaled - input_scaled, axis=1)
#
#     # Get top 5 most similar entries
#     top_indices = distances.argsort()[:5]
#     top_crops = data.iloc[top_indices]['label'].value_counts().index.tolist()
#
#     # Format output
#     result = {
#         'top_recommendation': top_crops[0],
#         'all_recommendations': [
#             {'crop': crop, 'probability': f"{(len(top_crops) - i) / len(top_crops):.0%}"}
#             for i, crop in enumerate(top_crops)
#         ]
#     }
#     return result