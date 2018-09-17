import sys
sys.path.append("../../utils/")

import os
import numpy as np
from PIL import Image
import pandas as pd
from pandas import HDFStore
from collections import defaultdict
# Simple RL
from simple_rl.tasks.navigation.NavigationStateClass import NavigationWorldState
# Google Maps
import google_maps_wrapper as MapsGoogle
# Utils
import geospatial_utils as GSUtil


def upsample(series, resample_rate, upsample_rate):
    """Resample Pandas series at @resample_rate and upsample @upsample_rate."""
    interpolated = series.resample(resample_rate).interpolate(method="linear")
    return interpolated.resample(upsample_rate).mean()


def upsample_df(df, columns, resample_rate='S', upsample_rate='5S'):
    """Upsample specified columns in the data frame."""
    df_cols = []
    for c in df:
        if c in columns:
            df_cols.append(upsample(df[c], resample_rate, upsample_rate))
        else:
            df_cols.append(df[c])

    resampled_df = pd.concat(df_cols, axis=1)
    return resampled_df.reindex(columns=df.columns).reset_index()


def compute_distance_and_heading(df):
    """Compute intersample distance and heading direction."""
    diff = df.diff(1)
    dx, dy = diff["longitude"], diff["latitude"]
    return np.sqrt(dy**2 + dx**2), np.arctan2(dy, dx)


class GeoLifeData(object):

    action_names = {0: "Stay", 1: "E", 4: "NE", 3: "N", 8: "NW",
                    5: "W", 14: "SW", 9: "S", 10: "SE"}
    a_to_lat_lng_chg = {'Stay': (0, 0), 'E': (0, 1), 'NE': (1, 1),
                        'N': (1, 0), 'NW': (1, -1), 'W': (1, 0),
                        'SW': (-1, -1), 'S': (-1, 0), 'SE': (-1, 1)}

    def __init__(self, hdf_file_name,
                 location="Beijing", lat_span_miles=5, lng_span_miles=5,
                 transport_modes=None, n_lat_states=100, n_lng_states=100,
                 feature_params=dict(
                     img_size="128x128",
                     img_type="satellite",
                     img_zoom=18,
                     gmaps_api_key=""),
                 dataset_dir="../Data",
                 debug=False):

        self.hdf_file_name = hdf_file_name
        self.dataset_dir = dataset_dir
        self.n_lat_states, self.n_lng_states = n_lat_states, n_lng_states
        self.data = GeoLifeData.load(self.hdf_file_name, self.dataset_dir)

        # Select Interest Region
        self.data, self.boundary = GSUtil.select_region(
            self.data, location, lat_span_miles, lng_span_miles)

        # Change NaN transport mode to "N/A".
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
        self.transport_modes = np.unique(self.data.transport_mode)
        self.trip_ids = np.unique(self.data.trip_id)
        self.user_ids = np.unique(self.data.user_id)

        self.tmode_by_user = pd.DataFrame(
            self.data.groupby("user_id")["transport_mode"].value_counts())
        self.tmode_by_user.columns = ["counts"]
        self.tmode_by_user.reset_index(inplace=True)
        self.tmode_by_user_pivot = self.tmode_by_user.pivot(
            index="user_id", columns="transport_mode", values="counts")

        if transport_modes:
            self.data = self._select_transport_modes(transport_modes)

        # State space
        self.latitude_levels = np.linspace(self.boundary["lat_min"],
                                           self.boundary["lat_max"],
                                           n_lat_states)
        self.longitude_levels = np.linspace(self.boundary["long_min"],
                                            self.boundary["long_max"],
                                            n_lng_states)

        # Features
        self.feature_params = feature_params
        self.feature_params["data_dir"] = "./state_{}x{}_features/".format(
            self.n_lat_states, self.n_lng_states)
        self.img_file_prefix = MapsGoogle.get_image_file_prefix(
            self.feature_params)
        self.debug = debug

    def _set_gmaps_api_key(self, gmaps_api_key):
        self.feature_params["gmaps_api_key"] = gmaps_api_key

    def _select_transport_modes(self, transport_modes):

        # Select samples by transport mode
        select_user_ids = self.tmode_by_user_pivot[
            self.tmode_by_user_pivot[transport_modes].sum(
                axis=1) > 100].index.values
        return self.data[(
            self.data["user_id"].isin(select_user_ids)) &
            (self.data["transport_mode"].isin(transport_modes))]

    # -------------------- #
    # --- Trajectories --- #
    # -------------------- #
    def get_trajectories(self, min_traj_samples=100, smoothing_k=2400,
                         stop_velocity=1e-8):

        traj_states_list = []

        for trip_id, df_trip in self.data.groupby("trip_id"):
            df_trip_resampled = upsample_df(df_trip.set_index(
                "date_time"), ["latitude", "longitude"], 'S', 'S')
            a, b, c = self.split_traj_by_goals(
                df_trip_resampled,
                min_traj_samples=min_traj_samples,
                smoothing_k=smoothing_k,
                stop_velocity=stop_velocity,
                latitude_levels=self.latitude_levels,
                longitude_levels=self.longitude_levels)
            traj_states_list.extend(b)

        traj_actions_list = []
        traj_mdp_states_list = []
        for traj_states in traj_states_list:
            a_list, a_names = self.states_to_actions(traj_states)
            traj_actions_list.append(a_list)
            traj_mdp_states_list.append(
                [NavigationWorldState(*s) for s in traj_states])

        return traj_states_list, traj_mdp_states_list, traj_actions_list

    # ---------------- #
    # --- Dynamics --- #
    # ---------------- #
    def get_dynamics(self):

        T = {}
        for lat_idx, lat in enumerate(self.latitude_levels):
            for lng_idx, lng in enumerate(self.longitude_levels):
                s = NavigationWorldState(lat, lng)
                T[s] = {}
                for a in GeoLifeData.action_names.values():
                    T[s][a] = {}
                    dlat, dlng = GeoLifeData.a_to_lat_lng_chg[a]
                    if lat_idx + dlat < len(self.latitude_levels) and \
                            lng_idx + dlng < len(self.longitude_levels):
                        s_prime = NavigationWorldState(
                            self.latitude_levels[lat_idx + dlat],
                            self.longitude_levels[lng_idx + dlng])
                        T[s][a][s_prime] = 1.
                    else:
                        s_prime = s  # Consider boundary as wall
                        T[s][a][s_prime] = 1.
        return T

    # --------- #
    # -- PHI -- #
    # --------- #
    def precache_state_feature(self):
        self.img_file_prefix = MapsGoogle.download_state_features(
            self.latitude_levels, self.longitude_levels, self.feature_params)

    def phi(self, state):

        lat, lng = state[0], state[1]
        img_file = os.path.abspath(self.img_file_prefix + "{}_{}".format(
            lat, lng) + ".jpg")

        if os.path.exists(img_file):
            return np.asarray(Image.open(img_file))
        else:
            # if self.debug:
            print("Downloading feature at (lat={}, lng={})".format(
                lat, lng))
            zoom = self.feature_params["img_zoom"]
            size = self.feature_params["img_size"]
            maptype = self.feature_params["img_type"]
            api_key = self.feature_params["gmaps_api_key"]
            return MapsGoogle.request_image_by_lat_lng(
                lat, lng, zoom, size, maptype, api_key)[0]

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
                                df_traj.at[mask, "transport_mode"] = \
                                    row["transport_mode"]
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

    # Split trip into sub-goal trajectories
    @staticmethod
    def split_traj_by_goals(df_trip, min_traj_samples=100, smoothing_k=1200,
                            stop_velocity=1e-8, latitude_levels=None,
                            longitude_levels=None, debug=False):

        tdiff = df_trip["date_time"].diff(1) / np.timedelta64(1, 's')
        disp = np.sqrt(df_trip["latitude"].diff(1)**2 +
                       df_trip["longitude"].diff(1)**2)
        df_trip["velocity"] = disp / tdiff
        """
        Smooth over latitude and longitude so that v is 0 if we keep circling
        around the same point (We care more about where's the goal i.e., where
        expert has spent more time, not the velocity. In addition, because
        readings are inherently erroneous we want those errors to cancel out
        instead of accumulating and giving false impression of motion.)
        """
        disp = np.sqrt(
            df_trip["latitude"].rolling(smoothing_k).mean().diff(1)**2 +
            df_trip["longitude"].rolling(smoothing_k).mean().diff(1)**2)
        df_trip["velocity_smooth"] = disp / tdiff.rolling(smoothing_k).mean()
        # df["velocity_smooth2"] = df["velocity"].rolling(smoothing_k).mean()
        df_trip["traj_ind"] = ~np.signbit(
            np.abs(df_trip["velocity_smooth"])-stop_velocity)

        if latitude_levels is not None and longitude_levels is not None:
            df_trip['latitude_discrete'] = pd.cut(
                df_trip['latitude'], latitude_levels,
                labels=latitude_levels[:-1])
            df_trip['longitude_discrete'] = pd.cut(
                df_trip['longitude'], longitude_levels,
                labels=longitude_levels[:-1])

        start_idx = 0
        trajectories = []
        traj_states_list = []
        for idx in np.where(np.diff(df_trip["traj_ind"]))[0]:

            pos_edge = df_trip["traj_ind"].iloc[idx+1]
            if not pos_edge:
                if (idx - start_idx) > min_traj_samples:
                    if debug:
                        print(idx, "end:", df_trip.iloc[idx]["date_time"])
                    trajectories.append(df_trip[start_idx:idx].copy())
                    if latitude_levels is not None\
                            and longitude_levels is not None:
                        traj_states_list.append([tuple(x) for x in
                                                 df_trip[start_idx:idx][[
                                                     "latitude_discrete",
                                                     "longitude_discrete"
                                                 ]].squeeze().values])
            else:
                start_idx = idx
                if debug:
                    print(idx, "start: ", df_trip.iloc[start_idx]["date_time"])

        if len(df_trip) - start_idx > min_traj_samples:
            if debug:
                print(len(df_trip)-1, "last:",
                      df_trip.iloc[len(df_trip)-1]["date_time"])
            trajectories.append(df_trip[start_idx:].copy())
            if latitude_levels is not None and longitude_levels is not None:
                traj_states_list.append(
                    [tuple(x) for x in df_trip[
                        start_idx:][["latitude_discrete",
                                     "longitude_discrete"]].squeeze().values])
        if debug:
            df_trip["traj_ind"] = pd.to_numeric(
                df_trip["traj_ind"]).astype(np.int)
            df_trip.plot(y="velocity", style='.-', subplots=True, sharey=True)
            df_trip.plot(x="date_time", y="velocity_smooth", style='.-',
                         subplots=True, sharey=True)
            df_trip.plot(x="date_time", y="traj_ind", style='.-',
                         subplots=True, sharey=True)

        return trajectories, traj_states_list, df_trip

    # --------------- #
    # --- Actions --- #
    # --------------- #
    @staticmethod
    def states_to_actions(states_list):

        df_t = pd.DataFrame(states_list, columns=["lat", "lng"])
        df_t['lat_diff'] = df_t['lat'].diff(1)
        df_t['lng_diff'] = df_t['lng'].diff(1)
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
            lambda x: GeoLifeData.action_names[x] if not pd.isna(x) else "")
        return df_t["action_str"].values.tolist(), GeoLifeData.action_names
