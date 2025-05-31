import requests
import time
import sqlite3
import Constants
import logging
import os

key = Constants.API_KEY


# Ensure the DB and table exist
DB_PATH = "elevation_cache.db"


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS elevation_cache (
                lat TEXT,
                lon TEXT,
                elevation REAL,
                PRIMARY KEY (lat, lon)
            )
        ''')
        conn.commit()


init_db()


def calcElevation(lat, lon):
    lat_str = str(round(float(lat), 6))  # Normalize to 6 decimal places
    lon_str = str(round(float(lon), 6))

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # Check cache
        cursor.execute(
            'SELECT elevation FROM elevation_cache WHERE lat = ? AND lon = ?', (lat_str, lon_str))
        result = cursor.fetchone()
        print("elevation result:", result)

        if result[0] > 0.0:
            elevation = result[0]
            print(f"Cache hit for ({lat_str}, {lon_str}): {elevation} meters")
            return round(elevation, 2)

        # Not in cache — query Google API
        print("Elevation not found in Cache - querying Elevation API!")
        # url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={lat_str},{lon_str}&key={key}"
        # using gpxz free tier
        url = f"https://api.gpxz.io/v1/elevation/gmaps-compat/json?locations={lat_str},{lon_str}&key={key}"
        print(url, "this is the google api url")
        time.sleep(1)  # throttle API calls to meet requests limits

        json_response = requests.get(url).json()

        try:
            elevation = json_response['results'][0]['elevation']
        except (KeyError, IndexError):
            logging.warning(
                f"Invalid response from Google Elevation API: {json_response}")
            elevation = 0
            # raise RuntimeError(
            #    f"Invalid response from Google Elevation API: {json_response}")

        print("this is the elevation result from google earth:", elevation)

        # Store in cache
        cursor.execute('INSERT OR REPLACE INTO elevation_cache (lat, lon, elevation) VALUES (?, ?, ?)',
                       (lat_str, lon_str, elevation))
        conn.commit()

        return round(elevation, 2)
