import matplotlib.pyplot as plt

def plot_p_vals_all_states(p_vals, D_traj_states, D_traj_actions, legend=True):

    plt.figure(figsize=(20,8))
    for k,v in p_vals.items():
        plt.plot(v, label=k)
    plt.xlabel("Training Iters")
    plt.ylabel("P")
    plt.title(r"P($s$,$a$)")
    if legend:
        plt.legend(loc='best')

def plot_p_vals_traj_states(p_vals, D_traj_states, D_traj_actions, legend=True):

    plt.figure(figsize=(20,8))
    for t_i, traj in enumerate(D_traj_states):
        s_i = 0
        state = traj[s_i]
        t_s_a_key = str(state) + ", a: " + str(D_traj_actions[t_i][s_i])
        plt.plot(p_vals[t_s_a_key], label= r"$\tau_{}$: ".format(t_i) + t_s_a_key)
    plt.xlabel("Training Iters")
    plt.ylabel("P")
    plt.title(r"P($s$,$a$)")
    if legend:
        plt.legend(loc='best')
