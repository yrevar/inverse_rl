import math
import requests
import pandas as pd
from persistence import PickleWrapper

# Geopy
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="my-application")

# Distances are measured in miles.
# Longitudes and latitudes are measured in degrees.
# Earth is assumed to be perfectly spherical.
earth_radius = 3963.0
degrees_to_radians = math.pi/180.0
radians_to_degrees = 180.0/math.pi

geopy_results = PickleWrapper("./geopy_results.p")
query_to_geocode = geopy_results.load()


def get_location(query):
    # To limit usage, cache results.
    # https: // operations.osmfoundation.org/policies/nominatim/
    if query in query_to_geocode:
        geocode = query_to_geocode[query]
    else:
        geocode = geolocator.geocode(query)
        query_to_geocode[query] = geocode
        geopy_results.dump(query_to_geocode)
    return (geocode.latitude, geocode.longitude)


def gmaps_get_location(query):
    # Google APIs

    # Source: https://stackoverflow.com/questions/25888396/how-to-get-latitude-longitude-with-python
    response = requests.get(
        'https://maps.googleapis.com/maps/api/geocode/json?address='
        + query.replace(" ", "+"))
    resp_json_payload = response.json()
    loc = resp_json_payload['results'][0]['geometry']['location']
    return (loc["latitude"], loc["longitude"])


def change_in_latitude(miles):
    """Given a distance north, return the change in latitude.

    Source:
        https://www.johndcook.com/blog/2009/04/27/converting-miles-to-degrees-longitude-or-latitude/
    """
    return (miles/earth_radius)*radians_to_degrees


def change_in_longitude(latitude, miles):
    """Given a latitude and a distance west, return the change in longitude.

    Source:
        https://www.johndcook.com/blog/2009/04/27/converting-miles-to-degrees-longitude-or-latitude/
    """
    # Find the radius of a circle around the earth at given latitude.
    r = earth_radius*math.cos(latitude*degrees_to_radians)
    return (miles/r)*radians_to_degrees


def get_bbox(location=(0, 0), lat_span_miles=2, lng_span_miles=2):

    lat, long = location
    lat_diff = change_in_latitude(lat_span_miles) / 2.
    long_diff = change_in_longitude(lat, lng_span_miles) / 2.
    return lat - lat_diff, lat + lat_diff, long - long_diff, long + long_diff

def filter_region_points(data, address_query, lat_span_miles,
                  lng_span_miles, lat_col="latitude",
                  lng_col="longitude"):
    """Crop dataframe centered at @address_query.

    Returns:
        Filtered dataframe and bounding box coordinates.
    """
    lat, lng = get_location(address_query)
    lat_min, lat_max, long_min, long_max = get_bbox(
        (lat, lng), lat_span_miles, lng_span_miles)

    return data[(data[lat_col] > lat_min) & (data[lat_col] < lat_max) &
                (data[lng_col] > long_min) & (data[lng_col] < long_max)], dict(lat_min=lat_min, lat_max=lat_max, long_min=long_min, long_max=long_max)

def filter_region_trips(data, address_query, lat_span_miles,
                  lng_span_miles, lat_col="latitude",
                  lng_col="longitude", trip_col="trip_id"):
    """Crop dataframe centered at @address_query.

    Returns:
        Filtered dataframe and bounding box coordinates.
    """
    lat, lng = get_location(address_query)
    return df_crop_trips(data, lat, lng, lat_span_miles, lng_span_miles, lat_col, lng_col, trip_col)

def df_crop_trips(data, lat, lng, lat_span_miles,lng_span_miles, lat_col="latitude", lng_col="longitude", trip_col="trip_id"):
    
    lat_min, lat_max, long_min, long_max = get_bbox(
        (lat, lng), lat_span_miles, lng_span_miles)
    
    n_trips = 0
    r_trips = 0
    in_region_trips = []
    for trip_id, df_trip in data.groupby(trip_col):
        n_trips += 1
        if df_trip["latitude"].min() >= lat_min and df_trip["latitude"].max() <= lat_max \
            and df_trip["longitude"].min() >= long_min and df_trip["longitude"].max() <= long_max:
                r_trips += 1
                in_region_trips.append(df_trip)
    print("Trips cropped: {}/{}".format(r_trips, n_trips))
    return pd.concat(in_region_trips), dict(lat_min=lat_min, lat_max=lat_max, long_min=long_min, long_max=long_max)

