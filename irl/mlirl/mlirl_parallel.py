import os
import sys
import time
import pickle
import numpy as np
from collections import defaultdict
import matplotlib as mpl
# mpl.use('Agg')

# GeoLife
sys.path.append("../../dataset/GeolifeTrajectories1.3/")
import geolife_data as GLData

# Torch
import torch

from torch import nn
import torch.optim as optim
from torch.autograd import Variable
# import torch.multiprocessing as mp
import torch.multiprocessing as mp
# Other methids were painful, and didn't work on my machine.
torch.multiprocessing.set_start_method("fork", force=True)

# Linear features
class LinearRewardModel(nn.Module):
    
    def __init__(self, phi_dim):
        super().__init__()
        self.w = nn.Linear(phi_dim, 1, bias=True)
        self.relu = nn.ReLU()
        
    def forward(self, phi):
        return -self.relu(-self.w(phi)) # Force R to be non-positive
#         return self.w(phi)

def sched_kirkpatrick_cooling(e_max, e_min, cooling_alpha, size=1e10, endless=False):
    
    # init
    e = e_max
    for t in range(size):
        
        yield e
        new_e = e * cooling_alpha
        e = new_e if new_e > e_min else e_min
    
    if endless:
        while True:
            yield e
        
        
def sched_exp_decay(e_max, e_min, lam, size=1e10, endless=False):
    
    # init
    e = e_max
    for t in range(size):
                
        e = e_min + (e_max-e_min) * np.exp(-lam * t)
        yield e
        
    if endless:
        while True:
            yield e
        
def sched_half_life(e_max, e_min, half_life, size=1e10, endless=False):
    return sched_exp_decay(e_max=e_max, e_min=e_min, lam=np.log(2)/half_life, size=size, endless=endless)

def run_value_iteration(S, A, R, trans_fn, s_to_idx, boltzmann_temp, gamma, n_iters,
                        dtype, S_lat_to_lng, given_goal_idx=None, p_idx=None, 
                        verbose=False, converge_eps=1e-6):
    
    if verbose:
        print("Running VI x {}.. (goal idx: {})".format(
            n_iters, given_goal_idx), flush=True)
    
    # boltzmann_temp_schedule = list(sched_exp_decay(10., boltzmann_temp, lam=0.1, size=100))
    
    nS, nA = len(S), len(A)
    Pi = torch.ones(nS, nA, dtype=dtype) / nA
    V = -1 * torch.ones(nS, dtype=dtype)
    Q = -1 * torch.ones(nS, nA, dtype=dtype)
    
    # Given goal
    if given_goal_idx is not None:
        V[given_goal_idx] = 0
  
    # Value iteration
    for iterno in range(n_iters):
        
        V_copy = V.clone()
        for si, s in enumerate(S):
            
            # No need to compute value for terminal and given goal states.
            if si == given_goal_idx:
                continue
                
            for ai, a in enumerate(A):
                
                sp = trans_fn(s, a, S_lat_to_lng)
                if sp is None: # outside envelope
                    continue
                else: 
                    if verbose:
                        print(s, "-", "{:5s}".format(a), "->", sp,
                              "R: ", float(R[si]),
                              "V_sp: ", gamma, V[s_to_idx[sp]].clone(), flush=True)
                    Q[si, ai] = R[si] + gamma * 1. * V[s_to_idx[sp]].clone()
                    
            # Pi[si, :] = boltzmann_dist(Q[si, :].clone(), boltzmann_temp_schedule[iterno])
            Pi[si, :] = boltzmann_dist(Q[si, :].clone(), boltzmann_temp)
            V[si] = Pi[si, :].clone().dot(Q[si, :].clone())
        
        if torch.max(torch.abs(V-V_copy)) <= converge_eps:
            print("\n\t\t -> VI Converged at {} <-".format(iterno))
            break
            
    return Pi, V, Q

def compute_gradient(trajectory, S, S_lat_to_lng, phi, A, trans_fn, r_model, 
                     boltzmann_temp, gamma, n_vi_iter, dtype, w_grad_queue, p_idx,
                     done, bprop_lock, perf_debug=False):
    
    # if perf_debug: print("Running process: #{}..".format(p_idx), flush=True)
    n_tau_sa = 0
    s_list, a_list = trajectory
    
    print("\t\tRunning process: {}, traj len: {}, states: {}".format(p_idx, len(s_list), len(S)))
    
    s_to_idx = defaultdict(lambda: None)
    for idx, s in enumerate(S):
        s_to_idx[s] = idx
    a_to_idx = {a: idx for idx, a in enumerate(A)}

    if perf_debug:
        _vi_start = time.time()
    
    R = [r_model(phi(S[si])).type(dtype) for si in range(len(S))]
    
    
        
    Pi, V, Q = run_value_iteration(S, A, R, trans_fn, s_to_idx, boltzmann_temp,
                                   gamma, n_vi_iter, dtype, S_lat_to_lng,
                                   s_to_idx[s_list[-1]], None, p_idx)
    if perf_debug:
        print("\t\tGoal {} VI time: {:f}".format(
            p_idx+1, time.time()-_vi_start), flush=True)

    loss = 0
    for idx in range(len(s_list)-1):  # discard end decision
        n_tau_sa += 1
        
        s_idx = s_to_idx[s_list[idx]]
        a_idx = a_to_idx[a_list[idx]]
        assert s_idx is not None, "Process {}: expert state {} not in the envelope! "\
            "Check dynamics or if right envelope is selected." .format(
                p_idx, s_list[idx]) # Check dynamics or if right envelope is selected.
        loss -= torch.log(Pi[s_idx, a_idx])
     
    if perf_debug:
        _bkw_start = time.time()
        
    #bprop_lock.acquire()
    # print("Loss: ", loss)
    loss.backward()
    #bprop_lock.release()
    
    if perf_debug:
        print("\t\tBprop time: {:f}".format(
            time.time()-_bkw_start), flush=True)

#     w_grad_queue.put((loss.item(), n_tau_sa))
#     w_grad_queue.put(None)
#     done.wait()
    return loss.item(), n_tau_sa, Pi, V, Q

def run_mlirl_step_by_goals(trajectories, S_list, S_lat_to_lngs_list, phi, A, trans_fn, r_model,
                            boltzmann_temp, gamma, vi_iters, dtype,
                            perf_debug=False, max_processes=1, verbose=False):
    
    total_loss = 0
    n_tau_sa = 0
    first_result = True
    
    Pi_list, V_list, Q_list = [], [], []

    for i in range(int(np.ceil(len(trajectories)/max_processes))):

        nb_ended_workers = 0
        w_grad_queue = mp.Queue()
        producers = []
        trajectories_batch = trajectories[i*max_processes: (i+1)*max_processes]

        done = mp.Event()
        bprop_lock = mp.Lock()
        for p_idx, trajectory in enumerate(trajectories_batch):
            S = S_list[i*max_processes + p_idx]
            S_lat_to_lng = S_lat_to_lngs_list[i*max_processes + p_idx]
#             p = mp.Process(target=compute_gradient,
#                            args=(trajectory, A, trans_fn, phi, r_model, boltzmann_beta,
#                                  S, gamma, vi_iters, dtype,
#                                  w_grad_queue, i*max_processes + p_idx, done,
#                                  bprop_lock, perf_debug))
            if verbose:
                print("\tForking process #{}..".format(i*max_processes + p_idx))
            
            loss, n_tau, Pi, V, Q = compute_gradient(
                trajectory, S, S_lat_to_lng, phi, A, trans_fn, r_model, 
                boltzmann_temp, gamma, vi_iters, dtype,
                w_grad_queue, i*max_processes + p_idx, done,
                bprop_lock, perf_debug)
            
            Pi_list.append(Pi)
            V_list.append(V)
            Q_list.append(Q)
            print("\tProcess #{} loss {}".format(i*max_processes + p_idx, loss))
            total_loss += loss
            n_tau_sa += n_tau
#             p.start()
#             producers.append(p)

#         while nb_ended_workers != len(trajectories_batch):
#             worker_result = w_grad_queue.get()

#             if worker_result is None:
#                 nb_ended_workers += 1
#                 if verbose:
#                     print("\tFinished process #{}..".format(nb_ended_workers),
#                           flush=True)
#             else:
#                 if first_result:
#                     total_loss, n_tau_sa = worker_result
#                     first_result = False
#                 else:
#                     total_loss += worker_result[0]
#                     n_tau_sa += worker_result[1]
#         done.set()
#     total_loss, n_tau_sa = 1, 1
    return total_loss, n_tau_sa, Pi_list, V_list, Q_list

# Exploration policy
def boltzmann_dist(Q, temperature):
    return (Q/temperature).softmax(dim=0)
    
def MLIRL(
    trajectories, S_list, phi, A, trans_fn, S_lat_to_lngs_list,
    r_model, r_optimizer, n_iter=1, n_vi_iter=1,
    gamma=0.99, boltzmann_temp=1.,
    loss_eps=1e-3, max_goals=None, dtype=torch.float64,
    print_interval=1, debug=True, verbose=False, perf_debug=False,
    max_processes=1, plot_interval=1,
    results_dir="./__mlirl/", dec_model=None, iter_handler=None):
    
    # Book-keeping
    loss_history = []
    try:

        for _iter in range(n_iter):
            
            # mlirl iter tick
            _mlirl_iter_start = time.time()
            
            # Zero grads
            r_optimizer.zero_grad()
            
            # MLIRL given goal step
            total_loss, n_tau_sa, Pi_list, V_list, Q_list = run_mlirl_step_by_goals(
                trajectories[:max_goals], S_list[:max_goals], S_lat_to_lngs_list[:max_goals], 
                phi, A, trans_fn, r_model,
                boltzmann_temp, gamma, n_vi_iter, dtype,
                perf_debug, max_processes,
                verbose=verbose)
            loss = total_loss / float(n_tau_sa)
            loss_history.append(loss)
            
            if iter_handler and _iter==0:
                iter_handler(_iter, loss_history, r_model, Pi_list, V_list, Q_list, results_dir, dec_model)

            # mlirl iter tock
            if debug and (_iter % print_interval == 0 or _iter == n_iter-1):
                print(">>> Iter: {:04d}, loss: {:09.6f}, likelihood: {:02.4f}, CPU time: {:f}".format(
                    _iter, loss, np.exp(-loss), time.time()-_mlirl_iter_start))
                
            if loss_eps is not None and loss < loss_eps:
                print(">>> Iter: {:04d} Converged.\n\n".format(_iter))
                break
                
            # Gradient step
            r_optimizer.step()
            
            if iter_handler:
                iter_handler(_iter+1, loss_history, r_model, Pi_list, V_list, Q_list, results_dir, dec_model)
                
    # If interrupted, return current results
    except KeyboardInterrupt:
        iter_handler(_iter, loss_history, r_model, Pi_list, V_list, Q_list, results_dir, dec_model, training_interrupted=True)
        return r_model, loss_history, Pi_list, V_list, Q_list
    except:
        raise
        
    return r_model, loss_history, Pi_list, V_list, Q_list
