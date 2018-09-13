import math
## Geopy
from geopy.geocoders import Nominatim
geolocator = Nominatim()

"""
To converting miles to longitude/latitude, I'm using a simple approximation as mentioned in the blog by John D. Cook: https://www.johndcook.com/blog/2009/04/27/converting-miles-to-degrees-longitude-or-latitude/.
"""
# Distances are measured in miles.
# Longitudes and latitudes are measured in degrees.
# Earth is assumed to be perfectly spherical.
earth_radius = 3960.0
degrees_to_radians = math.pi/180.0
radians_to_degrees = 180.0/math.pi

def get_geocode(query):
    return geolocator.geocode("Beijing")

def change_in_latitude(miles):
    "Given a distance north, return the change in latitude."
    return (miles/earth_radius)*radians_to_degrees

def change_in_longitude(latitude, miles):
    "Given a latitude and a distance west, return the change in longitude."
    # Find the radius of a circle around the earth at given latitude.
    r = earth_radius*math.cos(latitude*degrees_to_radians)
    return (miles/r)*radians_to_degrees

def get_bbox(location=(0, 0), lat_span_miles=2, lng_span_miles=2):
    
    lat, long = location
    lat_diff = change_in_latitude(lat_span_miles) / 2.
    long_diff = change_in_longitude(lat, lng_span_miles) / 2.
    return lat - lat_diff, lat + lat_diff, long - long_diff, long + long_diff

def spatial_crop_dataframe(data, address_query, lat_span_miles, lng_span_miles, lat_col="latitude", lng_col="longitude"):
    """ Crops dataframe by spatial limits, returns filtered dataframe and bounding box
    """
    geo_area = get_geocode(address_query)
    lat_min, lat_max, long_min, long_max = get_bbox(
                    (geo_area.latitude, geo_area.longitude), lat_span_miles, lng_span_miles)
    return data[(data[lat_col] > lat_min) & (data[lat_col] < lat_max) & \
                (data[lng_col] > long_min) & (data[lng_col] < long_max)], (lat_min, lat_max, long_min, long_max)



