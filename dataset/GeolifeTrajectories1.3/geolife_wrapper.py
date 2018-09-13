import pandas as pd
from pandas import HDFStore


def read_trajectory_plt(file_path):
    """ Redas and parses trajectory "<trip_id>.plt" file as Pandas dataframe.
    """
    return pd.read_csv(file_path,
                       skiprows=6, usecols=[0, 1, 3, 4, 5, 6],
                       parse_dates={'date_time': [4, 5]},
                       infer_datetime_format=True,
                       header=None, names=["latitude", "longitude", "x",
                                           "altitude", "n_days", "date", "time"])


def read_trajectory_labels(file_path):
    """ Reads and parses trajectory "labels.txt" file as Pandas dataframe.
    """
    return pd.read_csv(file_path, parse_dates=[0, 1],
                       infer_datetime_format=True, sep='\t',
                       skiprows=1, header=None,
                       names=["start_time", "end_time", "transport_mode"])


def find_trajectory_dirs_with_labels(dataset_dir):
    """ Finds and returns trajectory directories that has "labels.txt" in it.
    """
    labeled_dirs = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file == "labels.txt":
                labeled_dirs.append(root)
    return labeled_dirs


def get_dataframe_grouped_by_user(traj_dir_with_labels, select_user=None, process_labels=False):
    """ Reads trajectory directories and returns pandas data frame grouped by user.
    """
    assert(isinstance(traj_dir_with_labels, list))
    data = []
    lat_min, lat_max = -90, 90
    long_min, long_max = 0, 180
    for traj_dir in traj_dir_with_labels:

        print("Processing ", traj_dir)
        labels_file = os.path.join(traj_dir, "labels.txt")
        df_labels = read_trajectory_labels(labels_file)
        t_dir = os.path.join(traj_dir, "Trajectory")
        user_id = traj_dir.split("/")[-1]

        if select_user is not None and user_id != select_user:
            continue

        for file in os.listdir(t_dir):

            if file.endswith(".plt"):
                df_traj = read_trajectory_plt(os.path.join(t_dir, file))
                df_traj.dropna(inplace=True)
                df_traj["transport_mode"] = np.nan
                df_traj["trip_id"] = file.split(".")[0]
                df_traj["user_id"] = user_id
                label_exists = not (df_traj.iloc[0]["date_time"] >
                                    df_labels.iloc[-1]["end_time"] or
                                    df_traj.iloc[-1]["date_time"]
                                    < df_labels.iloc[0]["start_time"])
                if label_exists:
                    for index, row in df_labels.iterrows():
                        mask = (df_traj['date_time'] >= row["start_time"]) & \
                            (df_traj['date_time'] <= row["end_time"])
                        if sum(mask) > 0:
                            df_traj.at[mask, "transport_mode"] = row["transport_mode"]
                data.append(df_traj)
    return pd.concat(data, ignore_index=True)


def get_geolife_data(dataset_name="/geolife_trajectories_labelled",
                     hdf_file_name='./geolife_data_parsed.h5',
                     process_labels=True):
    """ Reads geolife data grouped by user. On first call, it stores the data
    into an HDF store, which is used for faster later retrievals.
    """
    store = HDFStore(hdf_file_name)
    if dataset_name in store.keys():
        data = store[dataset_name]
    else:
        data = get_dataframe_grouped_by_user(
            find_trajectory_dirs_with_labels("./Data/", process_labels))
        store[dataset_name] = data
    store.close()
    return data


def upsample(series, resample_rate, upsample_rate):
    """ Resamples pd series and upsamples.
    Required when we have inconsistent samples in time.
    """
    interpolated = series.resample(resample_rate).interpolate(method="linear")
    return interpolated.resample(upsample_rate).mean()


def upsample_df(df, columns, resample_rate='S', upsample_rate='5S'):
    """ Upsamples specified columns of the data frame
    """
    df_cols = []
    for c in df:
        if c in columns:
            df_cols.append(upsample(df[c], resample_rate, upsample_rate))
        else:
            df_cols.append(df[c])

    resampled_df = pd.concat(df_cols, axis=1)
    return resampled_df.reindex(columns=df.columns).reset_index()


def get_heading_and_distance(df):
    diff = df.diff(1)
    dx, dy = diff["longitude"], diff["latitude"]
    return np.sqrt(dy**2 + dx**2), np.arctan2(dy, dx)
