import os
import numpy as np
from io import BytesIO
from PIL import Image
from urllib import request

"""
# Ref: https://stackoverflow.com/questions/7490491/capture-embedded-google-map-image-with-python-without-using-a-browser

url = "http://maps.googleapis.com/maps/api/staticmap?center=-30.027489,-51.229248&size=100x100&zoom=18&sensor=false&maptype=satellite&sensor=false"
buffer = BytesIO(request.urlopen(url).read())
image = Image.open(buffer)
plt.imshow(image)
"""

def request_image_by_query(query, zoom=18, size="100x100", maptype="satellite", api_key=""):
    """
    Adapted from: http://drwelby.net/gstaticmaps/
    center= lat, lon or address
    zoom= 0 to 21
    maptype= roadmap, satellite, hybrid, terrain
    language= language code
    visible= locations
    """
    url = "http://maps.googleapis.com/maps/api/staticmap?center={}&size={}&zoom={}&sensor=false&maptype={}&style=feature%3Aall%7Celement%3Alabels%7Cvisibility%3Aoff&key={}".format(query, size, zoom, maptype, api_key)
    img = Image.open(BytesIO(request.urlopen(url).read()))
    return np.array(img.convert("RGB")), url

def request_image_by_lat_lng(lat, lng, zoom=18,
                                size="100x100", maptype="satellite",
                                api_key=""):
    return request_image_by_query("{},{}".format(lat,lng), zoom, size, maptype, api_key)

def download_state_features(latitude_levels, longitude_levels, to_dir,
                            size="32x32", zoom=18, maptype="satellite",
                            api_key=""):
    data_dir = os.path.join(to_dir, "imgs_" + size)
    os.makedirs(data_dir, exist_ok=True)
    print("Download started...")
    for lat in latitude_levels:
        for lng in longitude_levels:

            img = request_image_by_lat_lng(lat, lng, zoom, size, maptype, api_key)[0]
            Image.fromarray(img).save(
                    os.path.join(data_dir,
                                 "satimg_zm_" + str(zoom) + "_sz_" + size +  "_latlng_" + str(lat) + "_" + str(lng) + ".jpg"))
    print("Download finished...")
    return True
