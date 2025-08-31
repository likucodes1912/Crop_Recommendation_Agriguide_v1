from flask import Flask, render_template, request
import requests

app = Flask(__name__,template_folder='template')

# Replace with your OpenWeatherMap API key
API_KEY = "65f7db79b97b88af39774a60c8263588"
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

@app.route('/', methods=['GET', 'POST'])
def index():
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
                print(weather_data)
    
    return render_template('sample.html', weather_data=weather_data)
# , weather_data=weather_data

if __name__ == '__main__':
    app.run(debug=True,port=3000)




# from flask import Flask, render_template, request, jsonify
# import requests
# from datetime import datetime

# app = Flask(__name__)

# API_KEY = "http://api.openweathermap.org/data/2.5/forecast?id=524901&appid=65f7db79b97b88af39774a60c8263588"  # Replace with your actual API key
# BASE_URL = "http://api.openweathermap.org/data/2.5/weather"
# FORECAST_URL = "http://api.openweathermap.org/data/2.5/forecast"

# def get_weather_data(city):
#     params = {
#         "q": city,
#         "appid": API_KEY,
#         "units": "metric"
#     }
#     response = requests.get(BASE_URL, params=params)
#     return response.json()

# def get_forecast_data(city):
#     params = {
#         "q": city,
#         "appid": API_KEY,
#         "units": "metric"
#     }
#     response = requests.get(FORECAST_URL, params=params)
#     return response.json()

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/weather', methods=['POST'])
# def weather():
#     city = request.form['city']
#     weather_data = get_weather_data(city)
#     forecast_data = get_forecast_data(city)
    
#     if weather_data["cod"] != "404" and forecast_data["cod"] != "404":
#         current_weather = {
#             "city": city,
#             "temperature": round(weather_data["main"]["temp"]),
#             "description": weather_data["weather"][0]["description"],
#             "icon": weather_data["weather"][0]["icon"],
#         }
        
#         forecast = []
#         for item in forecast_data["list"][::8]:  # Get data for every 24 hours
#             forecast.append({
#                 "date": datetime.fromtimestamp(item["dt"]).strftime("%Y-%m-%d"),
#                 "temperature": round(item["main"]["temp"]),
#                 "description": item["weather"][0]["description"],
#                 "icon": item["weather"][0]["icon"],
#             })
        
#         return jsonify({"current": current_weather, "forecast": forecast})
#     else:
#         return jsonify({"error": "City not found"}), 404

# if __name__ == '__main__':
#     app.run(debug=True)

