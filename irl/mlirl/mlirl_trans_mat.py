import time
import numpy as np
from collections import defaultdict
# Torch imports
import torch
from torch.autograd import Variable


def compute_policy(S, A, R, T, idx_to_s, gamma, n_iters,
                   expl_policy, dtype, given_goal_idx=None, perf_debug=False):

    nS, nA = len(S), len(A)
    # Policy
    Pi = torch.ones(nS, nA, dtype=dtype) / nA
    # Value
    V = R[:, 0].clone()
    # Q value
    Q = R.repeat(1, nA).clone()

    # Check if state is terminal (stop leaking values back to
    # non-goal state space)
    # Done here so as to improve performance.
    S_ = [si for si in S if not (
        idx_to_s[si].is_terminal() or given_goal_idx == si)]

    if given_goal_idx:
        V[given_goal_idx] = 0

    # Value iteration
    for _vi_iter in range(n_iters):
        for si in S_:
            for ai in A:
                Q[si, ai] = R[si].clone() + gamma * T[si][ai].dot(V.clone())
            Pi[si, :] = expl_policy(Q[si, :].clone())
            V[si] = Pi[si, :].clone().dot(Q[si, :].clone())
    return Pi, V, Q


def group_by_goals(traj_states_list, traj_actions_list):

    goal_to_traj_idx = defaultdict(lambda: defaultdict(lambda: []))

    for traj_idx, traj_states in enumerate(traj_states_list):
        goal_to_traj_idx[traj_states[-1]]["traj_states_list"].append(
            traj_states)
        goal_to_traj_idx[traj_states[-1]]["traj_actions_list"].append(
            traj_actions_list[traj_idx])
    return goal_to_traj_idx

# Modified to handle given goal scenario.


def MLIRL(
        # Problem
        traj_state_idxs_list, traj_action_idxs_list, T, phi, idx_to_s,
        # MLIRL
        optimizer_fn, n_iter=1, n_vi_iter=1, w_init_scheme=None,
        w_init=None, gamma=0.99, boltzmann_beta=1., loss_eps=1e-3,
        goal_is_given=False, max_goals=None, dtype=torch.float64,
        print_interval=1, debug=True, verbose=False, perf_debug=False):

    assert w_init_scheme in ["random", "zeros", "custom"]
    if perf_debug:
        _mlirl_start = time.time()

    # States
    S = range(T.shape[0])
    # n_sa_pairs = sum(len(tau_s) for tau_s in traj_states_list)

    # Actions (assumes all actions are available in each state)
    A = range(T.shape[1])

    T = torch.Tensor(T).type(dtype)

    # Features
    phi_S = torch.Tensor([phi(idx_to_s[si]) for si in S]).type(dtype)
    phi_dim = phi_S.shape[1]

    # Reward parameters
    if w_init_scheme == "custom":
        assert w_init is not None
        w = Variable(torch.Tensor(w_init).type(dtype), requires_grad=True)
    elif w_init_scheme == "zeros":
        w = Variable(torch.zeros(phi_dim, 1).type(dtype), requires_grad=True)
    elif w_init_scheme == "random" or w_init_scheme is None:
        w = Variable(torch.Tensor(phi_dim, 1).normal_(0, 0.01).type(dtype),
                     requires_grad=True)

    # Optimization params
    optimizer = optimizer_fn([w])

    # Exploration policy
    def expl_policy(Q): return (boltzmann_beta*Q).softmax(dim=0)

    # Book-keeping
    loss_history = []

    # Group by goals
    goal_to_trajectories = group_by_goals(traj_state_idxs_list,
                                          traj_action_idxs_list)

    Pi, V, Q = None, None, None

    if perf_debug:
        print("\t\tMLIRL init time: {:f}".format(time.time()-_mlirl_start),
              flush=True)

    try:

        for _iter in range(n_iter):

            if perf_debug:
                _mlirl_iter_start = time.time()

            optimizer.zero_grad()
            total_loss = 0.
            n_sa_pairs = 0

            # Reward Estimate
            """
            Ideally we'll compute here, but pytorch we get following error:
                Trying to backward through the graph a second time, but the
                buffers have already been freed.
                Specify retain_graph=True when calling backward the first time.
            We can either pass retain_graph=True in backward(), or recompute.
            """
            # R = torch.mm(phi_S, w)

            for goal_no, (goal_state_idx, D) in enumerate(
                    goal_to_trajectories.items()):

                if max_goals is not None and goal_no >= max_goals:
                    break

                if perf_debug:
                    _r_start = time.time()
                # Reward Estimate
                R = torch.mm(phi_S, w)

                # Policy
                Pi, V, Q = compute_policy(S, A, R, T, idx_to_s, gamma,
                                          n_vi_iter, expl_policy, dtype,
                                          given_goal_idx=goal_state_idx if goal_is_given else None,
                                          perf_debug=perf_debug)

                if perf_debug:
                    print("\t\tGoal {} VI time: {:f}".format(goal_no+1, time.time()-_r_start))

                if perf_debug:
                    _loss_start = time.time()
                loss = 0
                for traj_idx, traj_state_idxs in enumerate(D["traj_states_list"]):
                    for sample_idx, si in enumerate(traj_state_idxs[:-1]):
                        n_sa_pairs += 1
                        ai = D["traj_actions_list"][traj_idx][sample_idx]
                        loss -= torch.log(Pi[si, ai])
                if perf_debug:
                    print("\t\t {} (s,a) loss compute time: {:f}".format(
                        n_sa_pairs, time.time()-_loss_start), flush=True)

                if perf_debug:
                    _mlirl_bkw_start = time.time()
                loss.backward()
                total_loss += loss
                if perf_debug:
                    print("\t\tLoss bkw time: {:f}".format(
                        time.time()-_mlirl_bkw_start), flush=True)
                # nn.utils.clip_grad_norm(R.parameters(), 0.5)

            grad_l2 = w.grad.norm(2)
            loss_norm = total_loss / float(n_sa_pairs)

            if debug and (_iter % print_interval == 0 or _iter == n_iter-1):
                print("Iter: {:04d}, loss: {:09.6f}".format(_iter, loss_norm))
                if verbose:
                    print("\n\t\t     w: [{}]"
                          "\n\t\t    dw: [{}]"
                          "\n\t\t||dw||: {:+012.7f}".format(
                              ' '.join("{:+012.7f}".format(v) for v in w[:, 0]),
                              ' '.join("{:+012.7f}".format(v) for v in -w.grad[:, 0]),
                              grad_l2))

            if loss_eps is not None and loss_norm < loss_eps:
                print("Iter: {:04d} Converged.".format(_iter))
                break

            optimizer.step()
            loss_history.append(loss_norm.data.tolist())
            if perf_debug:
                print("MLIRL iter {} time: {:f}".format(
                    _iter, time.time()-_mlirl_iter_start), flush=True)
    # If interrupted, return current results
    except KeyboardInterrupt:
        if debug:
            print("Iter: {:04d} Training interrupted! "
                  "Returning current results' snapshot.".format(_iter))
        if Pi is None:
            if debug:
                print("Iter: {:04d} Training interrupted! No results produced..".format(_iter))
            raise
        return w, R, Pi, V, Q, loss_history

    return w, R, Pi, V, Q, loss_history


def convert_T_to_matrix(T):

    S = list(T.keys())
    A = list(T[S[0]].keys())
    s_to_idx = {s: s_idx for s_idx, s in enumerate(T.keys())}
    a_to_idx = {a: a_idx for a_idx, a in enumerate(A)}
    T_mat = np.zeros((len(S), len(A), len(S)))
    for s in S:
        s_idx = s_to_idx[s]
        for a in A:
            a_idx = a_to_idx[a]
            for s_prime, p in T[s][a].items():
                s_prime_idx = s_to_idx[s_prime]
                T_mat[s_idx][a_idx][s_prime_idx] = p
    return T_mat, s_to_idx, a_to_idx


def convert_Tau_to_idxs(traj_states_list, traj_actions_list,
                        s_to_idx, a_to_idx):

    traj_state_idxs_list = [[s_to_idx[s] for s in traj_states]
                            for traj_states in traj_states_list]
    traj_action_idxs_list = [[a_to_idx[a] for a in traj_actions]
                             for traj_actions in traj_actions_list]
    return traj_state_idxs_list, traj_action_idxs_list


def run_MLIRL(irl_problem, mlirl_params):

    T = irl_problem.get_dynamics()
    T_mat, s_to_idx, a_to_idx = convert_T_to_matrix(T)
    idx_to_s = {v: k for k, v in s_to_idx.items()}

    Tau = irl_problem.sample_trajectories()
    traj_state_idxs_list, traj_action_idxs_list = convert_Tau_to_idxs(
        *Tau, s_to_idx, a_to_idx)

    def phi(s_idx): return irl_problem.features(s_idx)
    return MLIRL(traj_state_idxs_list, traj_action_idxs_list, T_mat, phi,
                 idx_to_s, **mlirl_params)
