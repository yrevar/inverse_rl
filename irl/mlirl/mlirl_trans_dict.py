import time
from collections import defaultdict
# Torch imports
import torch
from torch.autograd import Variable


def compute_policy(S, A, R, trans_dict, s_to_idx, gamma, n_iters,
                   expl_policy, dtype, given_goal_idx=None,
                   perf_debug=False):

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
    S_ = [s for s_idx, s in enumerate(S) if not (
        s.is_terminal() or given_goal_idx == s_idx)]

    if given_goal_idx:
        V[given_goal_idx] = 0

    # Value iteration
    for _vi_iter in range(n_iters):
        for s_idx, s in enumerate(S_):
            for a_idx, a in enumerate(A):
                for s_prime in trans_dict[s][a]:
                    Q[s_idx, a_idx] = R[s_idx].clone()
                    + gamma * trans_dict[s][a][s_prime] *\
                        V[s_to_idx[s_prime]].clone()
            Pi[s_idx, :] = expl_policy(Q[s_idx, :].clone())
            V[s_idx] = Pi[s_idx, :].clone().dot(Q[s_idx, :].clone())
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
        traj_states_list, traj_actions_list, trans_dict, phi,
        # MLIRL
        optimizer_fn, n_iter=1, n_vi_iter=1, w_init_scheme=None,
        w_init=None, gamma=0.99, boltzmann_beta=1., loss_eps=1e-3,
        goal_is_given=False, max_goals=None, dtype=torch.float64,
        print_interval=1, debug=True, verbose=False, perf_debug=False):

    assert w_init_scheme in ["random", "zeros", "custom"]
    if perf_debug:
        _mlirl_start = time.time()

    # States
    S = list(trans_dict.keys())
    s_to_idx = {s: idx for idx, s in enumerate(S)}
    # n_sa_pairs = sum(len(tau_s) for tau_s in traj_states_list)

    # Actions (assumes all actions are available in each state)
    A = list(trans_dict[S[0]].keys())
    a_to_idx = {a: idx for idx, a in enumerate(A)}

    # Features
    phi_S = torch.Tensor([phi(s) for s in S]).type(dtype)
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
    goal_to_trajectories = group_by_goals(traj_states_list, traj_actions_list)

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
            n_sa_pairs = 0.

            # Reward Estimate
            """
            Ideally we'll compute here, but pytorch we get following error:
                Trying to backward through the graph a second time, but the
                buffers have already been freed.
                Specify retain_graph=True when calling backward the first time.
            We can either pass retain_graph=True in backward(), or recompute.
            """
            # R = torch.mm(phi_S, w)

            for goal_no, (goal_state, D) in enumerate(
                    goal_to_trajectories.items()):

                if max_goals is not None and goal_no >= max_goals:
                    break

                if perf_debug:
                    _r_start = time.time()
                # Reward Estimate
                R = torch.mm(phi_S, w)

                # Policy
                Pi, V, Q = compute_policy(S, A, R, trans_dict, s_to_idx, gamma,
                                          n_vi_iter, expl_policy, dtype,
                                          given_goal_idx=s_to_idx[goal_state] if goal_is_given else None,
                                          perf_debug=perf_debug)

                if perf_debug:
                    print("\t\tGoal {} VI time: {:f}".format(
                        goal_no+1, time.time()-_r_start))

                if perf_debug:
                    _loss_start = time.time()
                loss = 0
                for traj_idx, traj_states in enumerate(D["traj_states_list"]):
                    for sample_idx, s in enumerate(traj_states[:-1]):
                        n_sa_pairs += 1
                        a_idx = a_to_idx[D["traj_actions_list"][traj_idx][sample_idx]]
                        loss -= torch.log(Pi[s_to_idx[s], a_idx])
                        # loss = -torch.log(Pi[s_to_idx[s], a_idx])
                if perf_debug:
                    print("\t\t {} (s,a) loss compute time: {:f}".format(
                        n_sa_pairs, time.time()-_loss_start), flush=True)

                if perf_debug:
                    _mlirl_bkw_start = time.time()
                loss.backward()
                total_loss += loss
                print("total_loss", total_loss)
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


def run_MLIRL(irl_problem, mlirl_params):

    traj_states, traj_actions = irl_problem.sample_trajectories()
    T = irl_problem.get_dynamics()

    def phi(s): return irl_problem.features(s)
    return MLIRL(traj_states, traj_actions, T, phi, **mlirl_params)
