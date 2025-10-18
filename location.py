import requests

city = input("What city are you currently in?")
api_key = "4e514b49d73362c5d739f05fea7f27cd"

url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={api_key}"

response = requests.get(url).json()

if response:
    lat = response[0]['lat']
    lon = response[0]['lon']
    print(f"City: {city}")
    print(f"Latitude: {lat}, Longitude: {lon}")
else:
    print("City not found!")