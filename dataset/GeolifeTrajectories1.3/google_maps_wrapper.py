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


def request_np_image_by_query(query, zoom=18, size="100x100",
                           maptype="satellite", api_key="", mode="RGB"):
    """
    Adapted from: http://drwelby.net/gstaticmaps/
    center= lat, lon or address
    zoom= 0 to 21
    maptype= roadmap, satellite, hybrid, terrain
    language= language code
    visible= locations
    """
    url = "http://maps.googleapis.com/maps/api/staticmap?center={}&size={}&zoom={}&sensor=false&maptype={}&style=feature%3Aall%7Celement%3Alabels%7Cvisibility%3Aoff&key={}".format(
        query, size, zoom, maptype, api_key)
    img = Image.open(BytesIO(request.urlopen(url).read()))
    return np.array(img.convert(mode)), url


def request_np_image_by_lat_lng(lat, lng, zoom=18,
                             size="100x100", maptype="satellite",
                             api_key="", mode="RGB"):
    return request_image_by_query("{},{}".format(lat, lng),
                                  zoom, size, maptype, api_key, mode)

def request_image_by_query(query, zoom=18, size="100x100",
                           maptype="satellite", api_key=""):
    """
    Adapted from: http://drwelby.net/gstaticmaps/
    center= lat, lon or address
    zoom= 0 to 21
    maptype= roadmap, satellite, hybrid, terrain
    language= language code
    visible= locations
    """
    url = "http://maps.googleapis.com/maps/api/staticmap?center={}&size={}&zoom={}&sensor=false&maptype={}&style=feature%3Aall%7Celement%3Alabels%7Cvisibility%3Aoff&key={}".format(
        query, size, zoom, maptype, api_key)
    img = Image.open(BytesIO(request.urlopen(url).read()))
    return img, url


def request_image_by_lat_lng(lat, lng, zoom=18,
                             size="100x100", maptype="satellite",
                             api_key=""):
    return request_image_by_query("{},{}".format(lat, lng),
                                  zoom, size, maptype, api_key)

def get_image_file_prefix(feature_params):

    data_dir = feature_params["data_dir"]
    size = feature_params["img_size"]
    maptype = feature_params["img_type"]
    zoom = feature_params["img_zoom"]

    store_dir = os.path.join(data_dir, "imgs_" + size)
    return os.path.join(store_dir, maptype[:3] + "img_zm_" + str(zoom) +
                        "_sz_" + size + "_latlng_")


def store_img(img, file):

    dirpath = os.path.dirname(file)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    Image.fromarray(img).save(file)


def download_state_features(latitude_levels, longitude_levels, feature_params):

    size = feature_params["img_size"]
    maptype = feature_params["img_type"]
    zoom = feature_params["img_zoom"]
    api_key = feature_params["gmaps_api_key"]
    file_prefix = get_image_file_prefix(feature_params)
    store_dir = os.path.dirname(file_prefix)

    os.makedirs(store_dir, exist_ok=True)
    print("Downloading images @ {}_<lat>_<lng>.jpg".format(file_prefix))
    for lat in latitude_levels:
        for lng in longitude_levels:

            img_file = file_prefix + str(lat) + "_" + str(lng) + ".jpg"

            if os.path.exists(img_file):
                print("Skipping download. File exists: {} ".format(img_file))
            else:
                img = request_image_by_lat_lng(lat, lng,
                                               zoom, size, maptype, api_key)[0]
                store_img(img, img_file)
    print("Download finished...")
    return os.path.abspath(store_dir)
