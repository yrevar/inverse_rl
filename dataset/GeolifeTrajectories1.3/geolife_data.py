import sys
sys.path.append("../../utils/")
import plotting_wrapper as Plotting
from PriorityQueue import PriorityQueue

import os
import numpy as np
from PIL import Image
import pandas as pd
from pandas import HDFStore
from collections import defaultdict

# Google Maps
import google_maps_wrapper as MapsGoogle
# Utils/
import geospatial_utils as GSUtil

import seaborn as sns
import matplotlib.pyplot as plt 

def resample(series, resample_rate):
    """Resample Pandas series at @resample_rate."""
    interpolated = series.resample(resample_rate).interpolate(method="linear")
    return interpolated #.resample(resample_rate).mean()

def upsample_df(df, columns, resample_rate='S'):
    """Upsample specified columns in the data frame."""
    df_cols = []
    column_names = []
    for c in df:
        if c in columns:
            df_cols.append(df[c])
            column_names.append(c + "_orig")
            df_cols.append(resample(df[c], resample_rate))
            column_names.append(c)
        else:
            df_cols.append(df[c])
            column_names.append(c)
    resampled_df = pd.concat(df_cols, axis=1, join="outer")
    resampled_df.columns = column_names
    return resampled_df.reindex().reset_index() #.ffill()

def discretize_trip(df_trip, lat_bins, lat_lvls, lng_bins, lng_lvls, 
                    resample_rate="L", drop_no_change=True, precision=12):

    df_trip_resampled = upsample_df(df_trip.copy().set_index(
                    "date_time"), ["latitude", "longitude"], resample_rate)
    df_trip_resampled['latitude_discrete'] = pd.cut(
        df_trip_resampled['latitude'], lat_bins, labels=lat_lvls, precision=precision)
    df_trip_resampled['longitude_discrete'] = pd.cut(
        df_trip_resampled['longitude'], lng_bins, labels=lng_lvls, precision=precision)

    # Drop (s,a) pairs with no change.
    if drop_no_change:        
        df_trip_resampled.drop(index=df_trip_resampled[
            (df_trip_resampled['latitude_discrete'].diff(1).shift(-1) == 0) &
            (df_trip_resampled['longitude_discrete'].diff(1).shift(-1) ==0)].index, inplace=True)
    return df_trip_resampled

def df_trip_to_trajectory(df_trip, lat_bins, lat_lvls, lng_bins, lng_lvls, 
                          resample_rate="L", drop_no_change=True):

    df = discretize_trip(df_trip, lat_bins, lat_lvls, lng_bins, lng_lvls, resample_rate, drop_no_change)
    s_list = list(zip(df.latitude_discrete, df.longitude_discrete))
    a_list, a_names = states_to_4actions(s_list)
    return s_list, a_list, a_names, df

def df_get_lat_lng_span(df, lat_col="latitude", lng_col="longitude"):
    lat_min, lat_max = df[lat_col].dropna().min(), df[lat_col].dropna().max()
    lng_min, lng_max = df[lng_col].dropna().min(), df[lng_col].dropna().max()
    return lat_min, lat_max, lng_min, lng_max

def get_lat_lng_span(s_list, lat_res=0, lng_res=0):
    s_list = np.asarray(s_list)
    return s_list[:,0].min()-lat_res, s_list[:,0].max()+lat_res, s_list[:,1].min()-lng_res, s_list[:,1].max()+lng_res

def select_geolife_trips1(geolife):
    """
    Uses simple heuristic to filter out unwanted ones. Removes all
    trajectories with inter-sample delay > 10 seconds to get the best trajectories. 
    Only 16 are left, but it works for now.
    """
    selected_trip_ids = []
    for trip_id, df_trip in geolife.get_trips():
        if np.sum(df_trip["date_time"].diff(1) / np.timedelta64(1, 's') > 10) < 1:
            selected_trip_ids.append(trip_id)
    return selected_trip_ids

def convert_to_trajectories(geolife, trip_ids, resample_rate="10L", drop_no_change=True):
    
    trajectories = []
    
    for trip_id, df_trip in geolife.get_trips():
        if trip_id in trip_ids:
    
            trajectories.append(
                df_trip_to_trajectory(df_trip,
                                      geolife.latitude_bins, 
                                      geolife.latitude_levels, 
                                      geolife.longitude_bins, 
                                      geolife.longitude_levels, 
                                      resample_rate, drop_no_change) + (df_trip,))
    return trajectories

def trans_func(state, action,
               lat_resolution=0.00010952786669320442,
               lng_resolution=0.00014278203208415482,
               st_fp_precision=20,
               a_to_lat_lng_chg = 
               {'Stay': (0, 0), 'E': (0, 1), 'NE': (1, 1),
                    'N': (1, 0), 'NW': (1, -1), 'W': (0, -1),
                    'SW': (-1, -1), 'S': (-1, 0), 'SE': (-1, 1)}):

    lat, lng = state
    lat = round(lat, st_fp_precision)
    lng = round(lng, st_fp_precision)

    lat_chg, lng_chg = a_to_lat_lng_chg[action]

    return round(lat + lat_chg * lat_resolution, st_fp_precision), \
            round(lng + lng_chg * lng_resolution, st_fp_precision)

# --------- #
# -- PHI -- #
# --------- #
def feature(state, store_dir="./features/satellite/", img_type="satellite",
            zoom=19, img_size="64x64", mode="L", api_key=None, verbose=False):
    
    lat, lng = state
    
    if store_dir:
        store_dir = os.path.join(store_dir, "imgs_" + img_size)
        os.makedirs(store_dir, exist_ok=True)
    
        img_file = os.path.join(store_dir, 
                        "map_{}_zm_{}_sz_{}_m_{}_latlng_{:+024.020f}_{:+025.020f}.jpg".format(
                            img_type[:3], zoom, img_size, mode, lat, lng))
        
        if os.path.exists(img_file):
            return np.asarray(Image.open(img_file).convert(mode))
        else:
            if verbose: print("Downloading {}".format(img_file))
            img = MapsGoogle.request_image_by_lat_lng(
                lat, lng, zoom, img_size, img_type, api_key)[0]
            img = np.asarray(img.convert(mode))
            MapsGoogle.store_img(img, img_file)
            return img
    else:
        img = MapsGoogle.request_image_by_lat_lng(
                lat, lng, zoom, img_size, img_type, api_key)[0]
        img = np.asarray(img.convert(mode))
        return img
    
def feature_cached(state, store_dir="./features/satellite/", img_type="satellite",
            zoom=19, img_size="64x64", mode="L"):
    
    lat, lng = state
    store_dir = os.path.join(store_dir, "imgs_" + img_size)
    img_file = os.path.join(store_dir, 
                    "map_{}_zm_{}_sz_{}_m_{}_latlng_{:+024.020f}_{:+025.020f}.jpg".format(
                        img_type[:3], zoom, img_size, mode, lat, lng))
    return os.path.exists(img_file)
            
# --------------- #
# --- Actions --- #
# --------------- #
def states_to_actions(states_list, include_stay=False):
    
    if include_stay:
        action_names = {0: "N", 1: "E", 4: "NE", 3: "N", 8: "NW", 
                         5: "W", 14: "SW", 9: "S", 10: "SE"}
    else:
        action_names = {0: "Stay", 1: "E", 4: "NE", 3: "N", 8: "NW",
                         5: "W", 14: "SW", 9: "S", 10: "SE"}
        
    df_t = pd.DataFrame(states_list, columns=["lat", "lng"])
    df_t['lat_diff'] = df_t['lat'].diff(1).shift(-1)
    df_t['lng_diff'] = df_t['lng'].diff(1).shift(-1)
    df_t["action"] = np.nan
    """
    (sign(lat), sign(lng))     -> Mapping
    (0, 0): stay               -> 0
    (0, 1): east               -> 1
    (1, 1): north east         -> 4
    (1, 0): north              -> 3
    (1, -1): north west        -> 8
    (0, -1): west              -> 5
    (-1, -1): south west       -> 14
    (-1, 0): south             -> 9
    (-1, 1): south east        -> 10

    action = wlatn * lat_neg + wlngn * lng_neg +
                        wlatp * lat_pos + wlngp * lng_pos
    wlngp, wlatp, wlngn, wlatn = 1, 3, 5, 9
    """
    df_t["action"] = df_t.apply(lambda row:
                                1 * np.int(row["lng_diff"] > 0) +
                                3 * np.int(row["lat_diff"] > 0) +
                                5 * np.int(row["lng_diff"] < 0) +
                                9 * np.int(row["lat_diff"] < 0), axis=1)

    df_t["action_str"] = df_t["action"].apply(
        lambda x: action_names[x] if not pd.isna(x) else "")
    
    return df_t["action_str"].values.tolist(), action_names
    
def states_to_4actions(states_list):
    
    lat_prev, lng_prev = states_list[0]
    a_list = []
    
    for lat_next, lng_next in states_list[1:]:
        
        lat_diff, lng_diff = lat_next - lat_prev, lng_next - lng_prev
        if lat_diff == 0 or abs(lat_diff) < abs(lng_diff):
            if lng_diff > 0:
                a_list.append("E")
            elif lng_diff < 0:
                a_list.append("W")
            else:
                a_list.append("Stay")
        elif lng_diff == 0 or abs(lat_diff) > abs(lng_diff):
            if lat_diff > 0:
                a_list.append("N")
            else: # lat_diff must be < 0 bc Stay we already checked
                a_list.append("S")
                
        lat_prev, lng_prev = lat_next, lng_next
    
    # Last action undefined
    a_list.append(None)
    return a_list, ["E", "W", "N", "S"]

from collections import defaultdict

def a_cost(a):
    return np.sqrt(2) if len(a) == 2 else 1

def a_cost_sum(a_list):
    cost = 0.
    for a in a_list:
        cost += a_cost(a)
    return cost

def path_cost(s_list, cost_fn):
    cost = 0.
    s = s_list[0]
    for sp in s_list[1:]:
        cost += cost_fn(s, sp)
        s = sp
    return cost

def heuristic_l2(start, goal, lat_step, lng_step):
    return np.sqrt((np.abs(goal[0]-start[0]) / lat_step)**2 + (np.abs(goal[1]-start[1]) / lng_step)**2)

def heuristic_l1(start, goal, lat_step, lng_step):
    return (np.abs(goal[0]-start[0]) / lat_step) + (np.abs(goal[1]-start[1]) / lng_step)

def find_enevelope_a_star(trajectory, actions, trans_func, lat_res, lng_res,
#                    g_fn=lambda a: a_cost(a),
                   g_fn=lambda p1, p2, lat_res, lng_res: heuristic_l2(p1, p2, lat_res, lng_res),
                   h_fn=lambda p1, p2, lat_res, lng_res: heuristic_l2(p1, p2, lat_res, lng_res),
                   cost_ubound=1., debug=True):
    
    s_list, a_list = trajectory
    start, goal = tuple(s_list[0]), tuple(s_list[-1])
    
    expert_cost = path_cost(s_list, lambda x,y: g_fn(x, y, lat_res, lng_res)) #np.sum(g_fn(a) for a in a_list[:-1])
#     expert_cost = np.sum(g_fn(a) for a in a_list[:-1])
    cost_max = cost_ubound * expert_cost
    
    frontier =  PriorityQueue()
    frontier.append((h_fn(start, goal, lat_res, lng_res), 0, start))
    explored = defaultdict(lambda: False)
    cost = defaultdict(lambda: np.float("inf"))
    envelope = []
   
    while frontier.size():
        
        f, g, s = frontier.pop()
        # if debug: print("{:.2f} ({:8d}), ".format(f, frontier.size()), end="")
        explored[s] = True
        envelope.append(s)
            
        if s == goal:
            continue
            
        for a in actions:
            sp = trans_func(s, a) #geolife.trans_func(NavigationWorldState(*s), a)
            if not explored[sp] and sp not in frontier:
#                 g_new = g + g_fn(a)
                g_new = g + g_fn(s, sp, lat_res, lng_res)
                f_new = g_new + h_fn(sp, goal, lat_res, lng_res)
                # print(g_new, f_new)
                if g_new < cost[sp] and f_new <= cost_max:
                    frontier.append((f_new, g_new, sp))
                    cost[sp] = g_new
                    
    return envelope, expert_cost

class GeoLifeData(object):
    
    def __init__(self, hdf_file_name="./geolife_data_parsed.h5",
                 location=(39.9059631, 116.391248), 
                 lat_span_miles=25, lng_span_miles=25,
                 state_gap_feet=20,
                 select_transport_modes=["car"],
                 min_user_samples=100,
                 dataset_dir="../Data",
                 st_fp_precision=20,
                 debug=False):
        
        self.hdf_file_name = hdf_file_name
        self.dataset_dir = dataset_dir
        self.location = location
        self.state_gap_feet = state_gap_feet
        self.lat_span_miles = lat_span_miles
        self.lng_span_miles = lng_span_miles
        self.select_transport_modes = select_transport_modes
        self.st_fp_precision = st_fp_precision
        
        self.lat_resolution = round(GSUtil.change_in_latitude(self.state_gap_feet * 1/5280), self.st_fp_precision)
        self.lng_resolution = round(GSUtil.change_in_longitude(self.location[0], self.state_gap_feet * 1/5280), self.st_fp_precision)
        self.lat_min, self.lat_max, self.lng_min, self.lng_max = GSUtil.get_bbox(
            self.location, self.lat_span_miles, self.lng_span_miles)
        
        # State space
        self.latitude_levels = np.arange(self.lat_min, self.lat_max, self.lat_resolution)
        self.longitude_levels = np.arange(self.lng_min, self.lng_max, self.lng_resolution)
        self.n_lat_states = len(self.latitude_levels)
        self.n_lng_states = len(self.longitude_levels)
        
        self.latitude_bins = np.hstack(
            (self.latitude_levels - self.lat_resolution / 2., 
             self.latitude_levels[-1] + self.lat_resolution / 2.))

        self.longitude_bins = np.hstack(
            (self.longitude_levels - self.lng_resolution / 2., 
             self.longitude_levels[-1] + self.lng_resolution / 2.))
        
        #         self.states = [(lat, lng) for lat in self.latitude_levels for lng in self.longitude_levels]
        #         self.lat_lng_grid_idxs = [(lat_idx, lng_idx)
        #                                   for lat_idx, _ in enumerate(self.latitude_levels)
        #                                   for lng_idx, _ in enumerate(self.longitude_levels)]
        #         self.states_to_idxs = {s: i for i, s in enumerate(self.states)}

        print("Resolution: lat: {}, lng: {}".format(self.lat_resolution, self.lng_resolution))
        print("No. states in {}x{} miles: {}x{}".format(self.lat_span_miles, self.lng_span_miles, self.n_lat_states, self.n_lng_states))
        print("Loading GeoLife data..")
        self.data = GeoLifeData.load(self.hdf_file_name, self.dataset_dir, 
                                     hdf5_data_name="/geolife_trajectories_labelled", process_labels=True)
        
        
        self.data = self.data.fillna({"transport_mode": "N/A"})
        # Sort by Date Time
        self.data = self.data.sort_values(by="date_time")
        """
        Some taxi samples have multiple samples recorded on single timestamp
        E.g.,
        2008-10-31 10:43:08	39.919698	116.349964	20081031101923	taxi
        2008-10-31 10:43:11	39.919694	116.349964	20081031101923	taxi
        2008-10-31 10:43:11	39.919693	116.349963	20081031101923	taxi
        2008-10-31 10:43:13	39.919690	116.349963	20081031101923	taxi
        """
        self.data.drop_duplicates(subset=["date_time"],
                                  keep="last", inplace=True)

        # stats
        self.transport_modes = np.unique(self.data.transport_mode)
        print("Transport modes: ", self.transport_modes)
        self.trip_ids = np.unique(self.data.trip_id)
        self.user_ids = np.unique(self.data.user_id)

        # filter by transport mode
        tmode_by_user = pd.DataFrame(
            self.data.groupby("user_id")["transport_mode"].value_counts())
        tmode_by_user.columns = ["counts"]
        tmode_by_user.reset_index(inplace=True)
        tmode_by_user_pivot = tmode_by_user.pivot(
            index="user_id", columns="transport_mode", values="counts")

        select_user_ids = tmode_by_user_pivot[
                    tmode_by_user_pivot[select_transport_modes].sum(
                        axis=1) > min_user_samples].index.values
        print("Selecting user ids: {}, transport modes: {}".format(select_user_ids, select_transport_modes))
        self.data_filtered = self.data[(self.data["user_id"].isin(select_user_ids)) &
                    (self.data["transport_mode"].isin(select_transport_modes))].copy()
        
        self.data_filtered_cropped, _ = GSUtil.df_crop_trips(
            self.data_filtered.copy(), self.location[0], self.location[1], self.lat_span_miles, self.lng_span_miles)
        
        
    def plot_scatter(self):
        g = sns.jointplot("longitude", "latitude", data=self.data_filtered_cropped, kind="hex", height=10)
        g.fig.suptitle("GeoLife: {} trajectories. \n lat: [{:.4f}, {:.4f}], lng: [{:.4f}, {:.4f}]".format(
            self.select_transport_modes, self.lat_min, self.lat_max, self.lng_min, self.lng_max))
        return g
    
    def get_trips(self):
        return self.data_filtered_cropped.groupby("trip_id")
    
    def get_trips_by_ids(self, trip_ids):
        return self.data_filtered_cropped[self.data_filtered_cropped["trip_id"].isin(trip_ids)].groupby("trip_id")
    
    # ---------------- #
    # ---- States ---- #
    # ---------------- #
    def get_states(self):
        
        for lat in self.latitude_levels:
            for lng in self.longitude_levels:
                yield (lat, lng)
                
    # ---------------- #
    # --- Dynamics --- #
    # ---------------- #
    def trans_func(self, state, action, a_to_lat_lng_chg = 
                   {'Stay': (0, 0), 'E': (0, 1), 'NE': (1, 1),
                        'N': (1, 0), 'NW': (1, -1), 'W': (0, -1),
                        'SW': (-1, -1), 'S': (-1, 0), 'SE': (-1, 1)}):
        
        return trans_func(state, action, self.lat_resolution, self.lng_resolution, slf.st_fp_precision, a_to_lat_lng_chg)
    
    # ----------------------- #
    # --- GeoLife Parsing --- #
    # ----------------------- #
    @staticmethod
    def parse_trajectory_file(file_path):
        """Parse trajectory file "<trip_id>.plt"."""
        return pd.read_csv(file_path,
                           skiprows=6, usecols=[0, 1, 3, 4, 5, 6],
                           parse_dates={'date_time': [4, 5]},
                           infer_datetime_format=True,
                           header=None, names=["latitude", "longitude", "x",
                                               "altitude", "n_days", "date",
                                               "time"])

    @staticmethod
    def parse_trajectory_labels(file_path):
        """Parse trajectory labels file "labels.txt"."""
        return pd.read_csv(file_path, parse_dates=[0, 1],
                           infer_datetime_format=True, sep='\t',
                           skiprows=1, header=None,
                           names=["start_time", "end_time", "transport_mode"])

    @staticmethod
    def find_dirs_with_labels(dataset_dir):
        """Find trajectory directories containing "labels.txt"."""
        labeled_dirs = []
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file == "labels.txt":
                    labeled_dirs.append(root)
        return labeled_dirs

    @staticmethod
    def get_dataframe_grouped_by_user(traj_dir_with_labels,
                                      select_user=None,
                                      process_labels=False):
        """Parse trajectories and return data frame grouped by user."""
        assert(isinstance(traj_dir_with_labels, list))
        data = []
        for traj_dir in traj_dir_with_labels:

            print("Processing ", traj_dir)
            labels_file = os.path.join(traj_dir, "labels.txt")
            df_labels = GeoLifeData.parse_trajectory_labels(labels_file)
            t_dir = os.path.join(traj_dir, "Trajectory")
            user_id = traj_dir.split("/")[-1]

            if select_user is not None and user_id != select_user:
                continue

            for file in os.listdir(t_dir):

                if file.endswith(".plt"):
                    df_traj = GeoLifeData.parse_trajectory_file(
                        os.path.join(t_dir, file))
                    df_traj.dropna(inplace=True)
                    df_traj["transport_mode"] = np.nan
                    df_traj["trip_id"] = file.split(".")[0]
                    df_traj["user_id"] = user_id
                    label_exists = not (df_traj.iloc[0]["date_time"] >
                                        df_labels.iloc[-1]["end_time"] or
                                        df_traj.iloc[-1]["date_time"]
                                        < df_labels.iloc[0]["start_time"])
                    if label_exists and process_labels:
                        for index, row in df_labels.iterrows():
                            mask = (df_traj['date_time'] >= row["start_time"])\
                                & (df_traj['date_time'] <= row["end_time"])
                            if sum(mask) > 0:
                                df_traj.at[mask, "transport_mode"] = row["transport_mode"]
                    data.append(df_traj)
        return pd.concat(data, ignore_index=True)

    @staticmethod
    def load(hdf_file_name, dataset_dir,
             hdf5_data_name="/geolife_trajectories_labelled",
             process_labels=True):
        """Parse geolife data grouped by user.

        Store the datatframe in HDF store to speed up subsequent retrievals.
        """
        store = HDFStore(hdf_file_name)
        if hdf5_data_name in store.keys():
            data = store[hdf5_data_name]
        else:
            dirs_with_labels = GeoLifeData.find_dirs_with_labels(dataset_dir)
            data = GeoLifeData.get_dataframe_grouped_by_user(
                dirs_with_labels, process_labels)
            store[hdf5_data_name] = data
        store.close()
        return data
