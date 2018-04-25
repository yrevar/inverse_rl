import numpy as np


def softmax(x1, x2):
    max_x = max(x1, x2)
    min_x = min(x1, x2)
    return max_x + np.log(1 + np.exp(min_x - max_x))

def StateLogPartitionFn(R, nS, nA, T, gamma, epsilon, phi):
    """
    Implements Algorithm 1 from Zeibart et al. thesis (2010)

    Softmax exploration:
        bair.berkeley.edu/blog/2017/10/06/soft-q-learning/
        home.deib.polimi.it/restelli/MyWebSite/pdf/rl5.pdf
        medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-7-action-selection-strategies-for-exploration-d3a97b7cceaf
    """
    V_soft = -1e20 * np.ones((nS))
    while True:
        V_soft_prime = np.asarray([phi(s) for s in range(nS)])
        for s in range(nS):
            for a in range(nA):
                V_soft_prime[s] = softmax(V_soft_prime[s], R[s] + gamma * T[s, a].dot(V_soft))
        if np.all(np.abs(V_soft_prime - V_soft) < epsilon):
            break
        V_soft = V_soft_prime.copy()
    return V_soft

def get_Q_soft(V_soft, R, nS, nA, T, gamma):

    Q = np.zeros((nS, nA))
    for s in range(nS):
        for a in range(nA):
            Q[s, a] = R[s] + gamma * T[s, a].dot(V_soft)
    return Q

def get_softmax_policy(R, nS, nA, T, gamma, epsilon, phi):
    V_soft = StateLogPartitionFn(R, nS, nA, T, gamma, epsilon, phi)
    Q_soft = get_Q_soft(V_soft, R, nS, nA, T, gamma)
    return np.exp(Q_soft.T-V_soft).T

def compute_feature_expectations(trajectories, state_idx_to_features, gamma, exclude_absorbing_state=True):

    """
    Computes and returns: μ(π) from trajectories
    The expectation is taken with respect to the random state sequence s0, s1, ... drawn by starting from a state s0 ∼ D,
    and picking actions according to unknown π.

    Input:
        trajectories: 3D numpy matrix of size M x T x 3, where M is number of trajectories,
                        T is fixed length of the trajectory (padded with last state if terminated early), 3 is for (s, a, r)
        state_idx_to_features: 2D numpy matrix mapping state index to its features of size N x K,
                        N is total # of known states and K is its feature dimensionality
    """
    nS, dimS = state_idx_to_features.shape
    feature_expectations = np.zeros((dimS,))

    for trajectory in trajectories:
        for i, (s_idx, a_idx, _) in enumerate(trajectory): # we ignore rewards because that's assumed to be unknown
            if exclude_absorbing_state and s_idx == nS-1:
                break
            feature_i = state_idx_to_features[s_idx]
            # Assuming R is a linear combination of known features of the state s, R(s) = w . Phi(s)
            feature_expectations += feature_i #* (gamma**i)
    return feature_expectations / trajectories.shape[0]

def compute_state_visitation_frequency(trajectories, nS, nA, dynamics_T, R, gamma, phi, vi_eps=1e-10, verbose=False):
    """
    Computes state visitation frequency using forward algorithm, uses DP algorithm from Zeibart et. al (2008).

    p(s, t+1) = sum_s' sum_a ( p(s', t-1) * p(a|s') * p(s|s', a))
    i.e., Prob. of being in state at time t+1 = sum over all prev state and actions (
                                            probability of being in prev state at time t-1 * probability of taking action a in prev state
                                            * transition dynamics (state | prev state, action))
    The problem is we don't know p(s) at any time, so we start from p(s0) which can be easily computed from seeing first state of trajectories
    and dividing by number of trajectories (M). Once we have p(s,0) for all s, we can build up p(s,t) using DP. Iteratively compute p(s,t+1)
    from previously computed p(s, t-1), all we need is policy that affects the state evolution along with dynamics for stochastic actions.

    Ref: DPV chapter 7 on DP
    """

    M, T, _ = trajectories.shape
    state_idxs = range(nS)
    action_idxs = range(nA)

    # compute stochastic policy
    policy = get_softmax_policy(R, nS, nA, dynamics_T, gamma, vi_eps, phi)
    if verbose: print("Stochastic Pi[a|s]: \n [N, E, S, W]"), print(policy), gw.disp_custom_grid([gw.actions_name[a] for a in policy.argmax(axis=1)], formatting=lambda x: str(x))

    init_state_idxs = trajectories[:,0,0] # take first visited state from all trajectories
    t0_visitation_count, _ = np.histogram(init_state_idxs, bins=range(nS+1)) # n bins will get counts for n-1 states
    mu_0_s = t0_visitation_count / float(M) # mu_0_s = p(s0 = s), consider s1 and mu_1 if you're a 1 based indexing fan
    mu_t_s = np.zeros((T, nS))
    mu_t_s[0, :] = mu_0_s

    for t in range(1,T):
        if verbose: print("Prob of being in states at t =", t-1)
        if verbose: gw.disp_custom_grid(mu_t_s[t-1, :], formatting=lambda x: "{:.3f}".format(x))
        for s_idx in state_idxs[:-1]:  # don't include absorbing state
            for a_idx in action_idxs:
                for s_prev_idx in state_idxs[:-1]: # don't include absorbing state
                    # print(s_prev_idx, "-", a_idx, "->", s_idx,  ":", mu_t_s[t-1, s_prev_idx], "-", policy[s_prev_idx][a_idx], "->", dynamics_T[s_prev_idx, a_idx, s_idx])
                    # visitation freq of s = sum over all s' and a (
                    # prob of being in prev state P(s') * action prob under given policy P(a|s') * probability of transitioning to s P(s',a,s))
                    mu_t_s[t, s_idx] += mu_t_s[t-1, s_prev_idx] * policy[s_prev_idx][a_idx] * dynamics_T[s_prev_idx, a_idx, s_idx]

    # Each t represents the probability of reaching a particular state at time step t in a trajectory
    # We can expect it to drop for states that are less rewarding, and stay stable or steadily increase
    # summing up along t gives us total state visitation frequency for max length of trajetory
    return mu_t_s.sum(axis=0)

def max_ent_irl(trajectories, state_idx_to_features, gamma, nA, dynamics_T, n_epochs, alpha, phi, vi_eps=1e-10, epoch_summary_rate=None):

    augmented_feature_matrix = state_idx_to_features

    M, T, _ = trajectories.shape
    nS, dimS = augmented_feature_matrix.shape

    # initialize weights
    theta = np.random.uniform(size=(dimS,)) #np.random.normal(size=(dimS,))

    # calculate expert feature expectations
    feature_expectations = compute_feature_expectations(trajectories, augmented_feature_matrix, gamma)

    # run gradient descent on theta to minimize difference between observed feature expectations and
    # state visitation frequency for current (evolving) reward (parameterized by theta and T) function
    for i in range(n_epochs):

        curr_R = augmented_feature_matrix.dot(theta)
        svf = compute_state_visitation_frequency(trajectories, nS, nA, dynamics_T, curr_R, gamma, phi, vi_eps, verbose=False)
        grad = feature_expectations - augmented_feature_matrix.T.dot(svf)

        if epoch_summary_rate is not None and ((i+1)%epoch_summary_rate == 0 or i==n_epochs-1):
            print(" Epoch {}, sum|grad|: {}".format(i+1, np.abs(grad).sum()))
        theta += alpha * grad

    R = augmented_feature_matrix.dot(theta).reshape((nS,))
    #R[-1] = 0 # ignore absorbing state reward
    return R
