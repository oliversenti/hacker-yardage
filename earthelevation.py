import requests
import json
import Constants 


key = Constants.API_KEY

def calcElevation(lat,lon):
    lat = str(lat)
    lon = str(lon)
    url = "https://maps.googleapis.com/maps/api/elevation/json?locations="+lat+","+lon+"&key="+key
    print(url, "this is the google api url")
    json_response = requests.get(url).json()
    elevation = json_response['results'][0]['elevation']
    print("this is the elevation result from google earth:", elevation)
    return round(elevation,2)
