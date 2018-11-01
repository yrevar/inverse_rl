import os
import time
import pickle
import numpy as np
import matplotlib as mpl
mpl.use('Agg')

# Torch
import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import torch.multiprocessing as mp
if mp.get_start_method() != "fork":
    mp.set_start_method('fork')
    
import matplotlib.pyplot as plt

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

def compute_gradient(tau_goal, S, s_to_idx, A, a_to_idx,
                          R, T, expl_policy, gamma, n_vi_iter, dtype, given_goal_idx,
                          w_grad_queue, p_idx, done, perf_debug=False):
    
    # if perf_debug: print("Running process: #{}..".format(p_idx), flush=True)
    n_tau_sa = 0
    
    if perf_debug:
        _vi_start = time.time()
    Pi, V, Q = run_value_iteration(S, A, R, T, s_to_idx, expl_policy, gamma, n_vi_iter, dtype, given_goal_idx)
    if perf_debug:
        print("\t\tGoal {} VI time: {:f}".format(
            p_idx+1, time.time()-_vi_start), flush=True)
        
    loss = 0
    for tau in tau_goal:
        s_list, a_list = zip(*tau)
        for idx in range(len(s_list)-1): # discard end decision
            n_tau_sa += 1
            s_idx = s_to_idx[s_list[idx]]
            a_idx = a_to_idx[a_list[idx]]
            loss -= torch.log(Pi[s_idx, a_idx])
            
    if perf_debug:
        _bkw_start = time.time()
    loss.backward()
    
    if perf_debug:
        print("\t\tLoss bkw time: {:f}".format(
                time.time()-_bkw_start), flush=True)
        
    w_grad_queue.put((loss.item(), n_tau_sa))
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
            
            p = mp.Process(target=compute_gradient, args=(tau_by_goals[goal], S, s_to_idx, 
                                                               A, a_to_idx, R, T, expl_policy, gamma, 
                                                               vi_iters, dtype, s_to_idx[goal] if given_goal else None, 
                                                               w_grad_queue, i*max_processes + p_idx, done, perf_debug))
            if verbose: print("Forking process #{}..".format(i*max_processes + p_idx))
            p.start()
            producers.append(p)

        while nb_ended_workers != len(process_goals):
            worker_result = w_grad_queue.get()
            
            if worker_result is None:
                nb_ended_workers += 1
                if verbose: print("Finished process #{}..".format(nb_ended_workers), flush=True)
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
        traj_by_goals, trans_dict, phi_S, model,
        # MLIRL
        optimizer, n_iter=1, n_vi_iter=1, gamma=0.99, boltzmann_beta=1., loss_eps=1e-3,
        goal_is_given=False, max_goals=None, dtype=torch.float64,
        print_interval=1, debug=True, verbose=False, perf_debug=False, max_processes=8,
        plot_interval=1, results_dir="./__mlirl/", convae=None, results_dict=None, debug_grads=False):

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
    
    Pi, V, Q = None, None, None
    
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
            total_loss, n_tau_sa = run_mlirl_step_by_goals(traj_by_goals, S, s_to_idx,
                                                           A, a_to_idx, model, R, trans_dict,
                                                           expl_policy, gamma, n_vi_iter, dtype, 
                                                           goal_is_given, perf_debug, max_goals, max_processes,
                                                           verbose=verbose)
            loss = total_loss / float(n_tau_sa)
            loss_history.append(loss)
            
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
                results_dict["loss_history"] = loss_history
                    
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
