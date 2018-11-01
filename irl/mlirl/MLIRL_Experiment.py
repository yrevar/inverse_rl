import os
import sys
import time
import copy
import pickle
from collections import defaultdict
import numpy as np

import matplotlib as mpl
mpl.use('Agg')

# Utils
sys.path.append("../../utils_python/")
from PersistentDictionary import PersistentDictionary
import shutil

# GeoLife
from IRL_GeoLifeWorld import IRL_GeoLifeWorld, group_by_goals

# Torch
import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import torch.multiprocessing as mp
if mp.get_start_method() != "fork":
    mp.set_start_method('fork')

# MLIRL
from mlirl_multicore import MLIRL

# ConvAE
sys.path.append("../../models/")
import ConvAE as ConvAE
from pprint import pprint

from IPython import display
# https://github.com/pandas-profiling/pandas-profiling/issues/68
import matplotlib.pyplot as plt

## Google Maps API Key
api_key = 'AIzaSyDuhzUh0iEBCywR4DneXX4zgOUayYbCft0'
CONFIG_FILE = None
EXP_FILE_BACKUP = True

mlirl_params = dict(
    
    # Constants
    n_iter = None,
    n_vi_iter = None,
    boltzmann_beta = None,
    dtype = torch.float,
    gamma = 0.95,
    loss_eps = -np.log(0.9),
    goal_is_given = True,
    print_interval = 1,
    perf_debug = True,
    debug = True,
    max_goals = 8,
    max_processes = 8,
    results_dir = None,
    
    # Model
    model = None,
    # Optimizer
    optimizer = None,
    # ConvAE (for decoding weights)
    convae = None,
    debug_grads = True,
    verbose = True
)

exp_config_dict = dict(
    exp_title="EXP_MLIRL",
    identifier_prefix="",
    identifier_suffix="",
    identifier_params=["n_vi_iter",
                       "boltzmann_beta",
                       "img_size",
                       "lr",
                       "weight_decay"],
    # MLIRL
    mlirl_params=mlirl_params,
    n_iter=2,
    n_vi_iter=1,
    boltzmann_beta=1.,

    # dataset
    geolife_hdf_file="../../dataset/GeolifeTrajectories1.3/geolife_data_parsed.h5",

    # states
    n_lat_states=100,
    n_lng_states=100,
    states="100x100",
    transport_modes=["car"],

    # features
    features_cache_dir="../../dataset/GeolifeTrajectories1.3/features",
    gmaps_api_key=None,
    img_zoom=18,
    img_size="128x128",
    img_type="satellite",
    input_dim=(1,128,128),
    random_features=False,
    img_mode="L",
    encode_features=True,
    rewrite_features=False,
    MU = 0.,
    STD = 1.,

    # training
    lr = 0.01,
    weight_decay=1e-4,
    optimizer_fn=lambda params, lr, weight_decay: optim.Adam(params, lr=lr, weight_decay=weight_decay),
    model_restore_file=None,

    # compute 
    use_data_parallel=False,

    # arch
    nw_depth=6,
    conv_k_size=16,
    fc_latent_multiplier=4,
    code_size=128,
    strided_conv_freq=3,
    convae_model_state="../../models/EXP_GEOLIFE_FEATURES_CONVAE_STRIDED/"
        "states=100x100$img_size=128x128$conv_k_size=16$fc_latent_multiplier=4$"
        "code_size=128$strided_conv_freq=3$lr=0.0001$weight_decay=1e-09/results/model_state_ae.pt",
)

def get_experiment_dirname(exp_config):
    
    os.makedirs(exp_config["exp_title"], exist_ok=True)
    
    exp_id = "".join("$" + str(param) + "=" + str(exp_config[param]) for param in exp_config["identifier_params"])[1:]
    if "identifier_prefix" in exp_config and exp_config["identifier_prefix"]:
        exp_id = exp_config["identifier_prefix"] + "$" + exp_id
    if "identifier_suffix" in exp_config and exp_config["identifier_suffix"]:
        exp_id += "$" + exp_config["identifier_suffix"]
        
    return os.path.join(exp_config["exp_title"], exp_id), exp_config["exp_title"] + "__" + exp_id

def get_encoded_features(phi, S, enc, dtype=torch.float):
    with torch.no_grad():
        phiS_enc = torch.stack([enc(torch.from_numpy(phi(s))) for s in S]).type(dtype)
    return phiS_enc

class LinearRewardModel(nn.Module):
    
    def __init__(self, phi_dim):
        super().__init__()
        self.w = nn.Linear(phi_dim, 1, bias=True)
        
    def forward(self, phi):
        return self.w(phi)
    
if __name__== "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Experiment configuration
    if CONFIG_FILE:
        
        EXP_CONFIG = PersistentDictionary(CONFIG_FILE, True)
        EXP_DIR, EXP_ID = get_experiment_dirname(EXP_CONFIG)
    else:   
        EXP_CONFIG = PersistentDictionary(None, verbose=True, **exp_config_dict)
        EXP_DIR, EXP_ID = get_experiment_dirname(EXP_CONFIG)
        EXP_CONFIG.write(os.path.join(EXP_DIR, "configs.p"))
        
    # Bring config parameters into scope
    globals().update(EXP_CONFIG)
    
    # Results
    RESULTS_DIR = os.path.join(EXP_DIR, "results")
    RESULTS_FILE = os.path.join(RESULTS_DIR, "results.p")
    RESULTS = PersistentDictionary(RESULTS_FILE, True).write()
    
    if EXP_FILE_BACKUP:
        exp_repro_file = "./{}/{}.py".format(exp_title, exp_title)
        os.makedirs(os.path.dirname(exp_repro_file), exist_ok=True)
        # shutil.copy(__file__, exp_repro_file)
    
    title_line = "{}".format(EXP_ID).replace("__", ": ").replace("$", ", ")
    title = title_line.replace(": ", ":\n\t").replace(", ", ",\n\t")
    print("{}\n{}\n{}".format("="*80, title, "="*80))
    
    # Architecture
    print("\nPreparing Net..")
    CONV_BLOCK = [("conv1", conv_k_size), ("relu1", None)]
    CONV_LAYERS = ConvAE.create_network(CONV_BLOCK, nw_depth, pooling_freq=1e100,
                                        strided_conv_freq=strided_conv_freq, strided_conv_channels=conv_k_size)
    CONV_NW = CONV_LAYERS + [("flatten1", None),
                             ("linear1", fc_latent_multiplier * code_size), ("linear1", code_size), ]

    print("NW Config (depth={}):\n\tBlock: ".format(len(CONV_NW)), end="")
    pprint(CONV_BLOCK)
    print("\tNet: \n", end="")
    pprint(CONV_NW)
    
    print("\nCreating model..")
    # https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530
    # torch.cuda.empty_cache() # Needed for repeated experiments
    convae = ConvAE.ConvAE(input_dim, 
                          enc_config=CONV_NW,
                           states_file=convae_model_state)
    if use_data_parallel and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        convae = nn.DataParallel(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\n\nUsing device {}".format(device))
    convae.to(device)
    
    # IRL Problem
    print("\nLoading IRL Problem..")
    irl_geolife = IRL_GeoLifeWorld(precache_state_feature=False,
                                   hdf_file_name=geolife_hdf_file,
                                   transport_modes=transport_modes, 
                                   feature_params=dict(
                                     img_size=img_size,
                                     img_type=img_type,
                                     img_zoom=img_zoom,
                                     gmaps_api_key=gmaps_api_key,
                                     cache_dir=features_cache_dir),
                                   n_lat_states=n_lat_states,
                                   n_lng_states=n_lng_states,
                                   debug=False)
    
    trajectories = irl_geolife.sample_trajectories()
    T = irl_geolife.get_dynamics()
    S = list(irl_geolife.T.keys())
    phi = lambda x, img_mode=img_mode: irl_geolife.features(x, img_mode, MU, STD)
    # Trajectories
    traj_by_goals = group_by_goals(trajectories)
    
    # Features
    print("\nPreparing features..")
    feature_cache_dir = os.path.dirname(convae_model_state)
    feature_file = os.path.join(feature_cache_dir, 
        "geolife_{}encoded_features_{}_states_{}_zoom_{}_size_{}_mode_{}_mu_{}_std_{}.pt".format(
        "" if encode_features else "not_", img_type, states, img_zoom, img_size, img_mode, MU, STD))
    
    if encode_features:
        encoding_fn = lambda x: convae.encode(x.float().view(-1,*input_dim)).squeeze()
    else:
        encoding_fn = lambda x: x

    if not os.path.exists(feature_file) or rewrite_features == True:
        phi_S = get_encoded_features(lambda x: irl_geolife.features(x, img_mode), S, encoding_fn)
        torch.save(phi_S, feature_file)
        print("Written {} features at {}".format(tuple(phi_S.shape), feature_file))
    else:
        phi_S = torch.load(feature_file)
        print("Loaded {} features from {}".format(tuple(phi_S.shape), feature_file))
    phi_dim = phi_S.shape[1]
    
    # Reward Model
    if model_restore_file is not None:
        with open(model_restore_file, "rb") as mlirl_model:
            print("Restoring model from {}".format(model_restore_file))
            M = pickle.load(mlirl_model)
    else:
        M = LinearRewardModel(phi_dim)

    for param in M.parameters():
        param.grad = Variable(torch.zeros(param.shape))
    # This is required for the ``fork`` method to work (https://pytorch.org/docs/master/notes/multiprocessing.html)
    M.share_memory()

    # Optimization params
    optimizer = optimizer_fn(M.parameters(), lr, weight_decay)
    
    # MLIRL params
    mlirl_params["model"] = M
    mlirl_params["optimizer"] = optimizer
    mlirl_params["convae"] = convae
    mlirl_params["n_iter"] = n_iter
    mlirl_params["n_vi_iter"] = n_vi_iter
    mlirl_params["boltzmann_beta"] = boltzmann_beta
    if mlirl_params["results_dir"] is None:
        mlirl_params["results_dir"] = RESULTS_DIR
        
    model, loss_history = MLIRL(traj_by_goals, T, phi_S, results_dict=RESULTS, **mlirl_params)
    RESULTS["loss_history"] = loss_history
    RESULTS.write()
