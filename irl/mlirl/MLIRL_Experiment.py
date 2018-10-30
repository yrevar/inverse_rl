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
sys.path.append("../../utils/")
from persistence import PickleWrapper
from persistence import ExperimentHelper
import shutil

# Plotting
sys.path.append("../../utils/")
import plotting_wrapper as Plotting

# IRL Interface
import sys
sys.path.append("../")
from IRLProblem import IRLProblem

# GeoLife Wrapper
sys.path.append("../../dataset/GeolifeTrajectories1.3/")
from geolife_data import GeoLifeData

# Torch
import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import torch.multiprocessing as mp
if mp.get_start_method() != "fork":
    mp.set_start_method('fork')
    
# ConvAE
import ConvAE as ConvAE
from ConvAE_Experiment import convae_solver

from IPython import display
# https://github.com/pandas-profiling/pandas-profiling/issues/68
import matplotlib.pyplot as plt

## Google Maps API Key
api_key = 'AIzaSyDuhzUh0iEBCywR4DneXX4zgOUayYbCft0'

class IRL_GeoLifeWorld(IRLProblem):
    
    # TODO: Cache Trajectories and Dyanmics?
    def __init__(self, precache_state_feature=False, encoder=None):
        self.geolife = GeoLifeData(hdf_file_name="../../dataset/GeolifeTrajectories1.3/geolife_data_parsed.h5",
                                   transport_modes=["car"], 
                                   feature_params=dict(
                                     img_size="64x64",
                                     img_type="satellite",
                                     img_zoom=18,
                                     gmaps_api_key=api_key,
                                     cache_dir="./features"),
                                   n_lat_states=100,
                                   n_lng_states=100,
                                   debug=False)
        if precache_state_feature:
            self.geolife.precache_state_feature()
        
        self.img_dim = tuple([int(x) for x in self.geolife.feature_params["img_size"].split("x")])
        
        self.phi = self.geolife.phi
        self.tau_s_list, self.tau_mdp_s_list, self.tau_a_list = None, None, None
        self.T = None
        self.encoder = encoder
        
    def write_features(self, S, file, dtype, mu, std, dont_encode=False):
        phi_S = torch.stack([self.features(s, mu, std, dont_encode) for s in S]).type(dtype)
        torch.save(phi_S, file)
        print("Done")
        
    def features(self, state, mu=0., std=1., dont_encode=False):
        img = torch.from_numpy(self.phi(state)).float().div(255.).sub(mu).div(std).view(1, *self.img_dim)
        
        if dont_encode is False:
            if self.encoder is not None:
                with torch.no_grad():
                    return self.encoder(img.view(-1, 1, *self.img_dim)).squeeze()
            else:
                raise ValueError("Encoder not known. Please set ``self.encoder.``")
        else:
            return img.squeeze()
    
    def sample_trajectories(self):
        if self.tau_s_list is None:
            self.tau_s_list, self.tau_mdp_s_list, self.tau_a_list, self.traj_dfs = self.geolife.get_trajectories()
        return self.tau_s_list, self.tau_mdp_s_list, self.tau_a_list
    
    def get_dynamics(self):
        if self.T is None:
            self.T = self.geolife.get_dynamics()
        return self.T
    
def group_by_goals(traj_states_list, traj_actions_list):
    
    goal_to_traj_idx = defaultdict(lambda: defaultdict(lambda: []))
    
    for traj_idx, traj_states in enumerate(traj_states_list):
        goal_to_traj_idx[traj_states[-1]]["tau_s_list"].append(traj_states)
        goal_to_traj_idx[traj_states[-1]]["tau_a_list"].append(traj_actions_list[traj_idx])
    return goal_to_traj_idx

def show_img(filename):
    plt.figure(figsize=(12,12))
    plt.imshow(plt.imread(filename))
    plt.axis('off')
    
def visualize_traj(tau_s_list, tau_a_list, idx):
    tr = np.asarray(tau_s_list[idx])
    a = tau_a_list[idx]
    
    lats = tr[:,0]
    lngs = tr[:,1]
    map_options = GMapOptions(lat=np.median(lats), lng=np.median(lngs), map_type="satellite", zoom=14)

    p = gmap(api_key, map_options, title="")

    source = ColumnDataSource(
        data=dict(lat=lats,
                  lon=lngs,
                  action=a)
    )
    p.text(x="lon", y="lat", text="action", text_font_size='8pt', text_color="red", source=source)
    show(p)
    # export_png(p, filename="./plot.png")
    # show_img("./plot.png")
    
def run_value_iteration(S, A, R, T, s_to_idx, expl_policy, gamma, n_iters,
                        dtype, given_goal_idx=None, verbose=False):

    nS, nA = len(S), len(A)
    
    # Initialize Pi, V, & Q.
    Pi = torch.ones(nS, nA, dtype=dtype) / nA
    V = R.clone() # trick to initialize terminal states' value without running VI on it (no need for absorbing state).
    Q = R.clone().reshape(nS, 1).repeat(1, nA)
    
    # Force given goals to have 0 value
    if given_goal_idx is not None:
        V[given_goal_idx] = 0
        
    if verbose: print("Running VI.. (given goal idx: {})".format(given_goal_idx), flush=True)
        
    # Value iteration
    for _ in range(n_iters):
        for si, s in enumerate(S):
            
            # No need to compute value for terminal and given goal states.
            if s.is_terminal() or si == given_goal_idx:
                continue 
                
            for ai, a in enumerate(A):
                for sp in T[s][a]:
                    if verbose: print(s, "-", "{:5s}".format(a), "->", sp, "R: ", float(R[si]), "V_sp: ", float(V[s_to_idx[sp]]))
                    Q[si, ai] = R[si].clone() + \
                        gamma * T[s][a][sp] * V[s_to_idx[sp]].clone()
            Pi[si, :] = expl_policy(Q[si, :].clone())
            V[si] = Pi[si, :].clone().dot(Q[si, :].clone())
    return Pi, V, Q

def compute_gradient(tau_s_list, tau_a_list, S, s_to_idx, A, a_to_idx,
                          R, T, expl_policy, gamma, n_vi_iter, dtype, given_goal_idx,
                          w_grad_queue, p_idx, done, perf_debug=False):
    
    # print("Process started: ", p_idx, flush=True)
    n_tau_sa = 0
    
    if perf_debug:
        _vi_start = time.time()
    Pi, V, Q = run_value_iteration(S, A, R, T, s_to_idx, expl_policy, gamma, n_vi_iter, dtype, given_goal_idx)
    if perf_debug:
        print("\t\tGoal {} VI time: {:f}".format(
            p_idx+1, time.time()-_vi_start), flush=True)
        
    loss = 0
    for tau_idx, tau_s in enumerate(tau_s_list):
        for sample_idx, s in enumerate(tau_s[:-1]):
            n_tau_sa += 1
            a_idx = a_to_idx[tau_a_list[tau_idx][sample_idx]]
            loss -= torch.log(Pi[s_to_idx[s], a_idx])
            
    if perf_debug:
        _bkw_start = time.time()
    loss.backward()
    
    if perf_debug:
        print("\t\tLoss bkw time: {:f}".format(
                time.time()-_bkw_start), flush=True)
        
    w_grad_queue.put((loss, n_tau_sa))
    w_grad_queue.put(None)
    done.wait()

def run_mlirl_step_by_goals(tau_by_goals, S, s_to_idx, A, a_to_idx, model, R, T, 
                            expl_policy, gamma, vi_iters, dtype, given_goal, 
                            perf_debug=False, max_goals=1, max_processes=8, verbose=False):

    total_loss = 0
    n_tau_sa = 0
    first_result = True
    goals = list(tau_by_goals.keys())[:max_goals]
    
    for i in range(int(np.ceil(len(goals)/max_processes))):
        
        nb_ended_workers = 0
        w_grad_queue = mp.Queue()
        producers = []
        process_goals = goals[i*max_processes: (i+1)*max_processes]
        
        done = mp.Event()
        for p_idx, goal in enumerate(process_goals):

            tau_s_list = tau_by_goals[goal]["tau_s_list"]
            tau_a_list = tau_by_goals[goal]["tau_a_list"]
            p = mp.Process(target=compute_gradient, args=(tau_s_list, tau_a_list, S, s_to_idx, 
                                                               A, a_to_idx, R, T, expl_policy, gamma, 
                                                               vi_iters, dtype, s_to_idx[goal] if given_goal else None, 
                                                               w_grad_queue, p_idx, done, perf_debug))
            if verbose: print("Forking process #{}..".format(i*max_processes + p_idx))
            p.start()
            producers.append(p)

        while nb_ended_workers != len(process_goals):
            worker_result = w_grad_queue.get()
            
            if worker_result is None:
                nb_ended_workers += 1
                if verbose: print("Finished process #{}..".format(nb_ended_workers), flush=True)
#                 for p in model.parameters():
#                     print("Grads: ", p.grad)
#                     break
            else:
                if first_result:
                    total_loss, n_tau_sa = worker_result
                    first_result = False
                else:
                    total_loss += worker_result[0]
                    n_tau_sa += worker_result[1]
        done.set()
        
    return total_loss, n_tau_sa

def MLIRL(
        # Problem
        tau_s_list, tau_a_list, trans_dict, phi_S, model,
        # MLIRL
        optimizer, n_iter=1, n_vi_iter=1, gamma=0.99, boltzmann_beta=1., loss_eps=1e-3,
        goal_is_given=False, max_goals=None, dtype=torch.float64,
        print_interval=1, debug=True, verbose=False, perf_debug=False, max_processes=8,
        plot_interval=1, results_dir="./__mlirl/", convae=None, exp_helper=None, debug_grads=False):

    if perf_debug:
        _mlirl_start = time.time()

    # States
    S = list(trans_dict.keys())
    s_to_idx = {s: idx for idx, s in enumerate(S)}
    
    # Actions (assumes all actions are available in each state)
    A = list(trans_dict[S[0]].keys())
    a_to_idx = {a: idx for idx, a in enumerate(A)}
    
    # Exploration policy
    def expl_policy(Q): return (boltzmann_beta*Q).softmax(dim=0)

    # Book-keeping
    loss_history = []

    # Group by goals
    tau_by_goals = group_by_goals(tau_s_list, tau_a_list)

    Pi, V, Q = None, None, None
    
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        
    if perf_debug:
        print("\t\tMLIRL init time: {:f}".format(time.time()-_mlirl_start))
        
    try:

        for _iter in range(n_iter):

            if perf_debug:
                _mlirl_iter_start = time.time()
                
            # Zero grads
            optimizer.zero_grad()
            
            # Current Reward estimate
            R = model(phi_S)
            
            if debug_grads and _iter == 0:
                for p in model.parameters():
                    print(" Params @ Iter {}"
                          "\n\t\t     R: [mu {}, std {}]"
                          "\n\t\t     w: [{}]"
                          "\n\t\t    dw: [{}]".format(_iter, R.mean(), R.std(),
                              ' '.join("{:+012.7f}".format(v) for v in p[0, :]),
                              ' '.join("{:+012.7f}".format(v) for v in p.grad[0, :]))
                         )
                    break
            
            # MLIRL given goal step
            total_loss, n_tau_sa = run_mlirl_step_by_goals(tau_by_goals, S, s_to_idx,
                                                           A, a_to_idx, model, R, trans_dict,
                                                           expl_policy, gamma, n_vi_iter, dtype, 
                                                           goal_is_given, perf_debug, max_goals, max_processes,
                                                           verbose=verbose)
            loss = total_loss / float(n_tau_sa)
            loss_history.append(loss.item())
            
            if debug and (_iter % print_interval == 0 or _iter == n_iter-1):
                print("\n\n Iter: {:04d}, loss: {:09.6f}, likelihood: {:02.4f}".format(_iter, loss, np.exp(-loss)))
                
            if loss_eps is not None and loss < loss_eps:
                print("\n\n Iter: {:04d} Converged.".format(_iter))
                break
            
            if debug_grads:
                for p in model.parameters():
                    print(" Params @ Iter {}"
                          "\n\t\t     R: [mu {}, std {}]"
                          "\n\t\t     w: [{}]"
                          "\n\t\t    dw: [{}]".format(_iter, R.mean(), R.std(),
                              ' '.join("{:+012.7f}".format(v) for v in p[0, :]),
                              ' '.join("{:+012.7f}".format(v) for v in p.grad[0, :]))
                         )
                    break
            
            if results_dir:
                with open(os.path.join(results_dir, "./mlirl_iter_{}_model.pkl".format(_iter)), "wb") as model_file:
                    pickle.dump(model, model_file)
                    
                with open(os.path.join(results_dir, "./mlirl_iter_loss.pkl"), "wb") as model_file:
                    pickle.dump(loss_history, model_file)
                    
            # Gradient step
            optimizer.step()
            
            if perf_debug:
                print("MLIRL iter {} time: {:f}".format(
                    _iter, time.time()-_mlirl_iter_start))
                
            if plot_interval and _iter % plot_interval == 0:
                
                fig = plt.figure(figsize=(20, 18))
                plt.subplot(211)
                
                if convae:
                    out = convae.decode(model.w.weight, None)
                    plt.imshow(out[0,0].data, cmap="gray")
                    plt.title("Max likelihood image")
                
                plt.subplot(212)
                plt.plot(np.exp(-np.asarray(loss_history)))
                plt.title("MLIRL: GeoLife Trajectories. Phi(s) = Google_Satellite_Image(s).")
                plt.xlabel("Iterations")
                plt.ylabel("Likelihood")
                if results_dir:
                    plt.savefig(os.path.join(results_dir, "mlirl_iter{}_plot.png".format(_iter)))
                plt.clf()
                
    # If interrupted, return current results
    except KeyboardInterrupt:
        print("Iter: {:04d} Training interrupted! "
                "Returning current results' snapshot.".format(_iter))
        return model, loss_history
    
    return model, loss_history

def read_encoded_features(file):
    try:
        phi_S = torch.load(file)
        print("Features loaded from file: {}\nShape: {}".format(file, phi_S.shape))
        return phi_S
    except FileNotFoundError:
        print("Feature encoding file {} doesn't exist!".format(file))
        raise

class LinearRewardModel(nn.Module):
    
    def __init__(self, phi_dim):
        super().__init__()
        self.w = nn.Linear(phi_dim, 1, bias=True)
        
    def forward(self, phi):
        return self.w(phi)

mlirl_params = dict(
    # Constants
    n_iter = 30,
    n_vi_iter = 100,
    dtype = torch.float,
    gamma = 0.95,
    boltzmann_beta = 1.,
    loss_eps = -np.log(0.9), # loss = -log(likelihood), 0.9 is a descent target for data likelihood/Product_i(P(s_i,a_i))
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

exp_config = dict(
    
    mlirl_config_params = mlirl_params,    
    #     state_features_file = "./EXP_FEATURES_CONVAE_NoImgNorm_NoPooling_StridedConv/results/ksize_16_fc_latent_512_code_128_depth_24_stridedconvfreq_3_lr_0.0001_wdecay_1e-09/features_state_100x100_unormalized_satimgs_64x64.pt",
    state_features_file="./EXP_FEATURES_CONVAE_NoImgNorm_NoPooling_StridedConv2/results/ksize_16_fc_latent_512_code_128_blocks6_stridedconvfreq_3_depth_17_lr_0.0001_wdecay_1e-09__acc_pearson_coeff/geolife_features_cache.pt",
#     model_state_file = "./EXP_FEATURES_CONVAE_NoImgNorm_NoPooling_StridedConv/results/ksize_16_fc_latent_512_code_128_depth_24_stridedconvfreq_3_lr_0.0001_wdecay_1e-09/model_state_ae.pt",
 model_state_file="./EXP_FEATURES_CONVAE_NoImgNorm_NoPooling_StridedConv2/results/ksize_16_fc_latent_512_code_128_blocks6_stridedconvfreq_3_depth_17_lr_0.0001_wdecay_1e-09__acc_pearson_coeff/model_state_ae.pt",
    MU = 0, #0.2963,
    STD = 1, #0.1986,
    input_dim = (1,64,64),
    lr = 0.01,
    weight_decay = 1e-4,
    optimizer_fn = lambda params, lr, weight_decay: optim.Adam(params, lr=lr, weight_decay=weight_decay),
    model_restore_file = None, #"./EXP_MLIRL_STRIDED_CONV2/geolife_results_vi_100_gamma_0.95_boltzmann_1.0_lr_0.01_wdecay_0/mlirl_iter_4_model.pkl",
    random_features = False,
)

if __name__== "__main__":
    
    np.random.seed(0)
    torch.manual_seed(0)
    
    EXPERIMENT = "EXP_MLIRL_STRIDED_CONV_NEW"
    EXP_DATA_FILE = "./{}/exp_data.p".format(EXPERIMENT)
    EXP_REPRO_FILE = "./{}/{}.py".format(EXPERIMENT, EXPERIMENT)
    print("Launching Experiment: {} \n\t configuration file: {} \n\t reproduction file: {}".format(
        EXPERIMENT, EXP_DATA_FILE, EXP_REPRO_FILE))

    os.makedirs(os.path.dirname(EXP_REPRO_FILE), exist_ok=True)
    shutil.copy(__file__, EXP_REPRO_FILE)

    exp = ExperimentHelper(exp_config, EXP_DATA_FILE)
    exp.launch(globals())
    
    # Model
    conv_block = [("conv1", 16), ("relu1", None)]
    nw = ConvAE.create_network(conv_block, 6, pooling_freq=1e100, 
                           strided_conv_freq=3, strided_conv_channels=16) + [
                    ("flatten1", None), ("linear1", 512), ("linear1", 128)]
    
    convae = ConvAE.ConvAE(input_dim, enc_config=nw, disable_decoder=False, states_file=model_state_file)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    
    # IRL Problem
    irl_geolife = IRL_GeoLifeWorld(encoder=None)
    tau_s_list, tau_mdp_s_list, tau_a_list = irl_geolife.sample_trajectories()
    T = irl_geolife.get_dynamics()
    
    # Encoded Features
    if random_features:
        print("Using randome features..")
        phi_S = torch.randn(10000, 128)
    else:
        print("Using features from: ", state_features_file)
        phi_S = read_encoded_features(state_features_file)
        
    phi_dim = phi_S.shape[1]

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
    
    mlirl_params["model"] = M
    mlirl_params["optimizer"] = optimizer
    mlirl_params["convae"] = convae
    if mlirl_params["results_dir"] is None:
        mlirl_params["results_dir"] = os.path.join(
            exp.data_dir, "geolife_results_vi_{}_gamma_{}_boltzmann_{}_lr_{}_wdecay_{}".format(
                mlirl_params["n_vi_iter"], mlirl_params["gamma"], mlirl_params["boltzmann_beta"], lr, weight_decay))

    model, loss_history = MLIRL(tau_mdp_s_list, tau_a_list, T, phi_S, exp_helper=None, **mlirl_params)
    print("loss:", loss_history)
