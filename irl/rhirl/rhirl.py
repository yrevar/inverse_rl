import time
import numpy as np
import tensorflow as tf
from collections import defaultdict
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState

# FLH: feature lookahead heap
def get_FLH(state, nA, phi, T, actions, h):

    frontier = [(state, 0, 0)] # for i, a in enumerate(range(len(actions)))] # BFS frontier: [(state, heap_idx, depth), ...]
    heap = [state] # for a in range(len(actions))]
    # parent_idx = [-1]
    depths = [0] # for a in range(len(actions))]
    idx = 0

    while len(frontier) != 0:
        state, heap_idx, d = frontier.pop(0)

        if d >= h:
            break

        for a in actions:
            state_prime = T(state, a)
            frontier.append((state_prime, idx+1, d+1))
            heap.append(state_prime)
            depths.append(d+1)
            # parent_idx.append(heap_idx)
            idx += 1

    features_depth = np.array(depths, np.int32)
    features_heap = np.asarray([phi(s) for s in heap], dtype=np.float32)
    """
    if feature heap is [s0, s01, s02, s03], FLH to layers will be
    [[0, s01, s02, s03], [s0, 0, 0, 0]]
    This is done to process feature tree layer wise from bottom to top. We start at bottom layer, and compute V, Q, and Pi; and then use those values to compute V,Q, and Pi of featues at layer above it.
    """
    heap_size = len(features_heap)
    FLH_to_layers = []
    phi_s_dim = len(features_heap[0])
    for i in range(h)[::-1]:
        start, end = int(nA*(1-nA**(i))/(1-nA))+1, int(nA*(1-nA**(i+1))/(1-nA))+1
        phi_i_tiled = np.repeat(features_heap[features_depth.squeeze() == i], nA, axis=0)
        phi_i_tiled_pad = np.vstack((np.zeros((start,phi_s_dim)), phi_i_tiled, np.zeros((heap_size-end,phi_s_dim))))
        FLH_to_layers.append(phi_i_tiled_pad)
    return FLH_to_layers, features_heap, features_depth

# For debugging purpose
def extract_action_subtree(feature_heap, H, nA, a_idx):
    layer = []
    idx = [] #(0, feature_heap[0])]
    for layer_idx in range(H+1):
        t_nodes = int((1-nA**(layer_idx+1))/(1-nA))
        t_nodes_prev = int((1-nA**(layer_idx))/(1-nA))
        l_nodes = t_nodes - t_nodes_prev
        la_nodes = int(l_nodes / nA)
        start_idx = int(t_nodes_prev + (la_nodes * a_idx))
        #print(t_nodes, l_nodes, la_nodes, start_idx, start_idx + la_nodes, list(range(start_idx, start_idx + la_nodes)))
        for i in range(start_idx, start_idx + la_nodes):
            #idx.append((layer_idx, feature_heap[i].astype(np.int).tolist()))
            idx.append(feature_heap[i])
            layer.append(layer_idx)
    return layer, np.asarray(idx)

def get_training_data(trajectories, phi, T, actions, h):

    D_FLH_layers = {}
    nA = len(actions)
    for t_i, traj in enumerate(trajectories):
        for s_i, state in enumerate(traj):
            if state not in D_FLH_layers:
                D_FLH_layers[state] = get_FLH(state, nA, phi, T, actions, h)[0]
    heap_size = len(D_FLH_layers[state][0])
    return D_FLH_layers, heap_size

def compute_cell_rewards(nvmdp, w_r, feature):
    r_map = np.zeros((nvmdp.height, nvmdp.width), dtype=np.float32)
    for row in range(nvmdp.height):
        for col in range(nvmdp.width):
            x, y = nvmdp._rowcol_to_xy(row, col)
            r_map[row, col] = feature(x, y).dot(w_r)[0]
    return r_map

def compute_cell_values(nvmdp, SFT_full, heap_size, nA, w_r, tf_graph):
    v_map = np.zeros((nvmdp.height, nvmdp.width), dtype=np.float32)
    for row in range(nvmdp.height):
        for col in range(nvmdp.width):
            x, y = nvmdp._rowcol_to_xy(row, col)
            v_map[row, col] = RHC_value(SFT_full[GridWorldState(x, y)], heap_size, nA, w_r, tf_graph)[0][0]
    return v_map

def RHC_policy(SFT, heap_size, nA, w_r, tf_graph):

    with tf.Session(graph=tf_graph) as sess:
        sess.run(tf.global_variables_initializer(), feed_dict={"w_init:0": w_r})
        pi = sess.run(fetches=["pi_out:0"],
                      feed_dict={"state_feature_tree:0": SFT,
                                 "pi_init:0": np.zeros((heap_size, nA), np.float32),
                                 "v_init:0": SFT[0].dot(w_r).squeeze()})
        return pi

def RHC_value(SFT, heap_size, nA, w_r, tf_graph):

    with tf.Session(graph=tf_graph) as sess:
        sess.run(tf.global_variables_initializer(), feed_dict={"w_init:0": w_r})
        v = sess.run(fetches=["v_out:0"],
                     feed_dict={"state_feature_tree:0": SFT,
                                "pi_init:0": np.zeros((heap_size, nA), np.float32),
                                "v_init:0": SFT[0].dot(w_r).squeeze()})
        return v

def RHC_rollout(nvmdp, state, phi, horizon, heap_size, w_r, tf_graph, max_traj_len):

    state_seq = [state]
    action_seq = []
    nA = len(nvmdp.actions)

    for i in range(max_traj_len):
        if (state.x, state.y) in nvmdp.goal_locs:
            break
        action = nvmdp.actions[np.argmax(\
                    RHC_policy(\
                        get_FLH(state, nA, phi, nvmdp.transition_func,\
                                nvmdp.actions, h=horizon)[0], heap_size, nA, w_r, tf_graph)[0][0])]
        state = nvmdp.transition_func(state, action)
        action_seq.append(action)
        state_seq.append(state)
    return state_seq, action_seq

def get_in_sample_data(n, D_traj_states, D_traj_actions, repetition=False, init_only=False):

    if init_only:
        return [traj[0] for traj in D_traj_states]

    if repetition is False:
        unq_states = list(set().union(*[set(traj) for traj in D_traj_states]))
        return [unq_states[idx] for idx in np.random.permutation(len(unq_states))]
    else:
        return [D_traj_states[i][np.random.randint(len(D_traj_states[i]))] for i in np.random.randint(0, len(D_traj_states), n)]

def run_rhc_test(nvmdp, n_test, phi, H, heap_size, w, tf_graph, max_traj_len=50):
    rhc_behavior = {}
    rhc_traj_ss, rhc_traj_as = [], []
    for init_state in nvmdp.sample_empty_states(n_test):
        states, actions = RHC_rollout(nvmdp, init_state, phi, H, heap_size, w, tf_graph, max_traj_len)
        for i in range(len(states)-1):
            rhc_behavior[states[i]] = actions[i]
        rhc_traj_ss.append(states)
        rhc_traj_as.append(actions)
    return rhc_traj_ss, rhc_traj_as, rhc_behavior

def run_rhc_train(nvmdp, D_traj_states, D_traj_actions, phi, H, heap_size, w, tf_graph, max_traj_len=50):
    rhc_behavior = {}
    rhc_traj_ss, rhc_traj_as = [], []
    for init_state in get_in_sample_data(len(D_traj_states), D_traj_states, D_traj_actions, init_only=True):
        states, actions = RHC_rollout(nvmdp, init_state, phi, H, heap_size, w, tf_graph, max_traj_len)
        for i in range(len(states)-1):
            rhc_behavior[states[i]] = actions[i]
        rhc_traj_ss.append(states)
        rhc_traj_as.append(actions)
    return rhc_traj_ss, rhc_traj_as, rhc_behavior

def build_rhirl_graph(heap_size, nA, phi_s_dim, gamma, h):

    g = tf.Graph()
    with g.as_default():
        def ComputeValue(W_r, V, nA, Pi, V_pad, tf_sft, i, bmann_beta=1.0):

            Phi_S = tf.gather(tf_sft, i)
            R = tf.matmul(Phi_S, W_r, name="R")
            V_next = tf.expand_dims(V.read(i, name="V_next"), -1)
            Q_h = tf.reshape(tf.squeeze(tf.add(R, tf.multiply(gamma, V_next), name="Q"))[1:], (-1, nA), "Q_sa_reshaped")
            Pi_h = tf.nn.softmax(Q_h, axis=-1, name="Pi_h")
            V_h = tf.squeeze(tf.pad(tf.reshape(
                    tf.reduce_sum(tf.multiply(Q_h, Pi_h), axis=-1, name="V"), [-1,1]), [[0, V_pad], [0, 0]], 'CONSTANT'))
            V = V.write(tf.add(i,1), V_h, name="V_update")
            i = tf.add(i, 1)
            return W_r, V, nA, Pi_h, V_pad, tf_sft, i

        def run_RHC(heap_size, nA, W_r, Pi, V_init, tf_sft, h):

            V = tf.TensorArray(tf.float32, size=0, dynamic_size=True,
                                     clear_after_read=False, infer_shape=False, name="V_array")
            V = V.write(0,  V_init, name="V_array_0")
            V_pad = tf.constant( int(( nA * (1-nA**(h))/(1-nA) ) - ( nA * (1-nA**(h-1))/(1-nA) )), dtype=tf.int32)
            loop_cond = lambda W_r, V, nA, Pi, V_pad, sft, i: tf.less(i, h, name="compute_value_end")
            W_r, V, nA, Pi, V_pad, sft, i = tf.while_loop(loop_cond, ComputeValue, [W_r, V, nA, Pi, V_pad, tf_sft, 0],
                                                      parallel_iterations=1, name="compute_value_loop")
            return V, Pi

        W_init = tf.placeholder(dtype=tf.float32, name="w_init", shape=(phi_s_dim, 1))
        Pi_init = tf.placeholder(dtype=tf.float32, name="pi_init") # this is completeley useless, it's only here due to lack of my TF skills
        V_init = tf.placeholder(dtype=tf.float32, name="v_init", shape=(heap_size)) # np.zeros(heap_size, dtype=np.float32)
        tf_sft = tf.placeholder(tf.float32, name="state_feature_tree")
        action_idx = tf.placeholder(tf.int32, name="action_idx")
        lr = tf.placeholder(tf.float32, name="learning_rate")
        W_r = tf.get_variable("w_r", dtype=tf.float32, initializer=W_init)
        V_out, Pi_out = run_RHC(heap_size, nA, W_r, Pi_init, V_init, tf_sft, h)
        log_likelihood = -tf.log(Pi_out[0, action_idx])
        W_grad = tf.gradients(log_likelihood, W_r)
        adam = tf.train.AdamOptimizer(learning_rate=lr)
        # adam = tf.train.GradientDescentOptimizer(learning_rate=lr)
        updateWeights = adam.apply_gradients(zip(W_grad, [W_r]), name="update_w_r")
        prob_summary = tf.summary.scalar('psa_summary', Pi_out[0, action_idx])
        Pi_out = tf.identity(Pi_out, name="pi_out")
        W_grad = tf.identity(W_grad, name="grad_w_r")
        # V_out_stack = tf.identity(V_out.stack(), name="v_out_stack")
        V_out = tf.identity(V_out.read(h), name="v_out")
        saver = tf.train.Saver()
    return g, saver

def rhirl_train(SFT, phi_s_dim, heap_size,
                nA, D_states, D_actions, tf_graph, saver=None,
                n_epochs=10, lr=0.1, write_summary=False, verbose=False, restore=False, run_rhc_callback=None, w_init=None):

    rhc_traj_ss_list = []

    if w_init is None:
        # w_init = np.zeros((phi_s_dim, 1)).astype(np.float32)
        w_init =np.random.uniform(0, 0.01, (phi_s_dim, 1)).astype(np.float32)
    with tf.Session(graph=tf_graph) as sess:

        if restore and saver is not None:
            saver.restore(sess, "/var/tmp/rhirl_model.ckpt")
        else:
            sess.run(tf.global_variables_initializer(),
                     feed_dict={"w_init:0": w_init})

        if write_summary:
            merged_summary = tf.summary.merge([prob_summary])
        p_vals = defaultdict(lambda: [])

        sft_time = 0.
        sft_ticks = 0.
        ep_time = 0.
        tot_time = 0
        cnt = 0

        tot_start = time.time()
        w = w_init
        for i in range(n_epochs):
            ep_start = time.time()
            if verbose: print("Epoch {}:".format(i))
            for t_i, traj in enumerate(D_states):

                for s_i, state in enumerate(traj[:-1]): # in last state no action is taken, so ignore

                    # if s_i is len(traj)-1:
                    #     action = np.random.randint(nA)
                    # else:
                    action = D_actions[t_i][s_i]
                    v_last = SFT[state][0].dot(w).squeeze()
                    # v_last = np.zeros(heap_size, dtype=np.float32)
                    feed_dict = {"state_feature_tree:0": SFT[state],
                                 "action_idx:0": action,
                                 "learning_rate:0":lr,
                                 "pi_init:0":np.zeros((heap_size, nA), np.float32),
                                 "v_init:0": v_last}

                    start = time.time()
                    pi, v_, w, grad, ugrad = sess.run(
                                fetches=["pi_out:0", "v_out:0", "w_r:0", "grad_w_r:0", "update_w_r"], feed_dict=feed_dict)
                    #print(v_)
                    sft_time += (time.time()-start)*1000
                    sft_ticks += 1
                    t_s_a_key = str(state) + ", a: " + str(action)
                    p_vals[t_s_a_key].append(pi[0, action])
                    if verbose: print("\t P({}, a: {}): {}".format(state, action, p_vals[t_s_a_key][-1]))

                if write_summary:
                    train_writer = tf.summary.FileWriter("/var/tmp/rhirl/summaries/train", sess.graph)
                    train_writer.add_summary(summary, cnt)
                    train_writer.close()
                cnt += 1

                # Prints P(first state, first action) of each trajectory
                # first_state_key = str(traj[0]) + ", a: " + str(D_actions[t_i][0])
                # print("\t P({}, a: {}): {:.4f}".format(traj[0], A_s[t_i][s_i], p_vals[first_state_key][-1]))

            ep_time += (time.time()-ep_start)*1000
            if verbose: print("Epoch time: ", (time.time()-ep_start)*1000)

            if run_rhc_callback:
                rhc_traj_ss, _, _ = run_rhc_callback(w)
                rhc_traj_ss_list.append(rhc_traj_ss)

        tot_time += (time.time()-tot_start)*1000
        if saver is not None:
            save_path = saver.save(sess, "/var/tmp/rhirl_model.ckpt")
            print("Model saved in path: %s" % save_path)

    return p_vals, w, sft_time / sft_ticks, ep_time / n_epochs, tot_time, rhc_traj_ss_list
