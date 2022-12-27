import requests
import json


apikey = "&key=AIzaSyAWDazVVwgKcinDG9geQnGFCMjVViPZ9pc"


def calcElevation(lat,lon):
    lat = str(lat)
    lon = str(lon)
    url = "https://maps.googleapis.com/maps/api/elevation/json?locations="+lat+","+lon+apikey
    print(url, "this is the gogole api url")
    json_response = requests.get(url).json()
    elevation = json_response['results'][0]['elevation']
    print("this is the elevation result from google earth:", elevation)
    return round(elevation)
