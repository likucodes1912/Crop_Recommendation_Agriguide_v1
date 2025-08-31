import datetime as dt
import requests

BASE_URL ="https://api.openweathermap.org/data/2.5/weather?"
API_KEY = open('api.txt','r').read()
# print(API_KEY)
# API_KEY = "65f7db79b97b88af39774a60c8263588"
CITY = str(input("Enter the CITY NAME: "))

def kelvin_to_celsius_fahrenheit(kelvin):
    celsius = kelvin - 273.15
    fahrenheit = celsius * (9/5) + 32
    return celsius, fahrenheit

url = BASE_URL + "appid=" + API_KEY + "&q=" + CITY

response= requests.get(url).json()
print(response)

temp_kelvin = response['main']['temp']
temp_celsius, temp_fahrenheit = kelvin_to_celsius_fahrenheit(temp_kelvin)
feels_like_kelvin = response['main']['feels_like']
feels_like_celsius, feels_like_fahrenheit = kelvin_to_celsius_fahrenheit(feels_like_kelvin)
wind_speed= response['wind']['speed']
humidity = response['main']['humidity']

description = response['weather'][0]['description']
sunrise_time = dt.datetime.utcfromtimestamp(response['sys']['sunrise'] + response['timezone'])
sunset_time = dt.datetime.utcfromtimestamp(response['sys']['sunset'] + response['timezone'])

print(f"{CITY} WEATHER REPORT TODAY.")
print(f"Temperature: {temp_celsius:.2f}°C or {temp_fahrenheit:.2f}°F")
print(f"Temperature feels like: {feels_like_celsius:.2f}°C or {feels_like_fahrenheit:.2f}°F")
print(f"Humidity:{humidity}%")
print(f"Wind Speed: {wind_speed}m/s")
print(f"Sun rises at {sunrise_time} local time.")
print(f"Sun sets at {sunset_time} local time.")

print(f"General Weather in {CITY}:{description}")


#TRY 2
# import requests
#
# api_key='65f7db79b97b88af39774a60c8263588'
# user_input= input("Enter City: ")
#
# # print(user_input)
#
# weather_data=requests.get(
#     f"https://api.openweathermap.org/data/2.5/weather?q={user_input}&units=imperial&appid={api_key}"
# )
#
# print(weather_data.status_code) # 200
# print(weather_data.json())
#
# # ERROR HANDLING IN USER_INPUT
# if weather_data.json()['cod'] == '404':
#     print("No City Found")
#
# else:
#     weather = weather_data.json()['weather'][0]['main']
#     temp = round(weather_data.json()['main']['temp'])
#
#     # print(f"Weather:"+weather,f"Temp:"+str(temp))
#
#     print(f"The Weather in {user_input} is:{weather}")
#     print(f"The Temperature in {user_input} is:{temp}°F")