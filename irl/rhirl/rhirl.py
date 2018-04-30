import time
import numpy as np
import tensorflow as tf
from collections import defaultdict

# FLH: feature lookahead heap
def get_FLH(state, phi, T, actions, h):

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
    FLH_to_layers = np.array([np.where(features_depth.reshape(-1,1) == i,
                                       features_heap,
                                       np.zeros_like(features_heap))
                                for i in range(h)][::-1])
    return FLH_to_layers, features_heap, features_depth

def get_training_data(trajectories, phi, T, actions, h):

    D_FLH_layers = {}
    for t_i, traj in enumerate(trajectories):
        for s_i, state in enumerate(traj[:-1]): # in last state no action is taken, so ignore
            if state not in D_FLH_layers:
                D_FLH_layers[state] = get_FLH(state, phi, T, actions, h)[0]
    heap_size = len(D_FLH_layers[state][0])
    return D_FLH_layers, heap_size

def compute_cell_rewards(nvmdp, w_r, feature):
    r_map = np.zeros((nvmdp.height, nvmdp.width), dtype=np.float32)
    for row in range(nvmdp.height):
        for col in range(nvmdp.width):
            x, y = nvmdp._rowcol_to_xy(row, col)
            r_map[row, col] = feature(nvmdp, x, y).dot(w)[0]
    return r_map

def build_rhirl_graph(heap_size, nA, phi_s_dim, gamma, h):

    g = tf.Graph()
    with tf.variable_scope("rhirl") as scope:
        with g.as_default():
            def ComputeValue(W_r, V, nA, Pi, V_pad, tf_sft, i):

                Phi_S = tf.gather(tf_sft, i)
                R = tf.matmul(Phi_S, W_r, name="R")
                V_prev = tf.expand_dims(V.read(i, name="V_prev"), -1) # For i=0, this will read hth step values which are 0
                Q_h = tf.reshape(tf.squeeze(tf.add(R, tf.multiply(gamma, V_prev), name="Q"))[1:], (-1, nA), "Q_sa_reshaped")
                Pi_h = tf.nn.softmax(Q_h, axis=-1, name="Pi_h")
                V_h = tf.squeeze(tf.pad(tf.reshape(
                        tf.reduce_sum(tf.multiply(Q_h, Pi_h), axis=-1, name="V"), [-1,1]), [[0, V_pad], [0, 0]], 'CONSTANT'))
                V = V.write(tf.add(i,1), V_h, name="V_update")
                i = tf.add(i, 1)
                return W_r, V, nA, Pi_h, V_pad, tf_sft, i

            def run_RHC(heap_size, nA, W_r, Pi, tf_sft, h):

                V = tf.TensorArray(tf.float32, size=0, dynamic_size=True,
                                         clear_after_read=False, infer_shape=False, name="V_array")
                V = V.write(0,  np.zeros(heap_size, dtype=np.float32), name="V_array_0")
                V_pad = tf.constant( int(( nA * (1-nA**(h))/(1-nA) ) - ( nA * (1-nA**(h-1))/(1-nA) )), dtype=tf.int32)

                loop_cond = lambda W_r, V, nA, Pi, V_pad, sft, i: tf.less(i, h-1, name="compute_value_end")
                W_r, V, nA, Pi, V_pad, sft, i = tf.while_loop(loop_cond, ComputeValue, [W_r, V, nA, Pi, V_pad, tf_sft, 0],
                                                          parallel_iterations=1, name="compute_value_loop")
                return V, Pi

            Pi_init = tf.placeholder(dtype=tf.float32, name="pi_init")
            tf_sft = tf.placeholder(tf.float32, name="state_feature_tree")
            action_idx = tf.placeholder(tf.int32, name="action_idx")
            lr = tf.placeholder(tf.float32, name="learning_rate")
            # W_r = tf.get_variable("W_r", dtype=tf.float32, initializer=tf.fill((phi_s_dim, 1), 0.)) # good prior
            W_r = tf.get_variable("w_r", dtype=tf.float32, initializer=tf.random_uniform((phi_s_dim, 1), 0, 0.1)) # better?
            V_out, Pi_out = run_RHC(heap_size, nA, W_r, Pi_init, tf_sft, h)
            log_likelihood = -tf.log(Pi_out[0, action_idx])
            W_grad = tf.gradients(log_likelihood, W_r)
            adam = tf.train.AdamOptimizer(learning_rate=lr)
            updateWeights = adam.apply_gradients(zip(W_grad, [W_r]), name="update_w_r")
            prob_summary = tf.summary.scalar('psa_summary', Pi_out[0, action_idx])
            Pi_out = tf.identity(Pi_out, name="pi_out")
            W_grad = tf.identity(W_grad, name="grad_w_r")
    return g

def rhirl_train(SFT, heap_size, nA, D_states, D_actions, tf_graph,
                n_epochs=10, lr=0.1, write_summary=False, verbose=False):

    with tf.Session(graph=tf_graph) as sess:

        sess.run(tf.global_variables_initializer())
        if write_summary:
            merged_summary = tf.summary.merge([prob_summary])
        p_vals = defaultdict(lambda: [])

        sft_time = 0.
        sft_ticks = 0.
        ep_time = 0.
        tot_time = 0

        cnt = 0
        tot_start = time.time()
        for i in range(n_epochs):
            ep_start = time.time()
            if verbose: print("Epoch {}:".format(i))
            for t_i, traj in enumerate(D_states):

                for s_i, state in enumerate(traj[:-1]): # in last state no action is taken, so ignore

                    feed_dict = {"state_feature_tree:0": SFT[state], "action_idx:0": D_actions[t_i][s_i],
                                         "learning_rate:0":lr, "pi_init:0": np.zeros((heap_size, nA), np.float32)}

                    start = time.time()
                    pi, w, grad, ugrad = sess.run(
                                fetches=["pi_out:0", "w_r:0", "grad_w_r:0", "update_w_r"], feed_dict=feed_dict)
                    sft_time += (time.time()-start)*1000
                    sft_ticks += 1
                    t_s_a_key = str(state) + ", a: " + str(D_actions[t_i][s_i])
                    p_vals[t_s_a_key].append(pi[0, D_actions[t_i][s_i]])
                    if verbose: print("\t P({}, a: {}): {}".format(state, D_actions[t_i][s_i], p_vals[t_s_a_key][-1]))

                if write_summary:
                    train_writer = tf.summary.FileWriter("./rhirl/summaries/train", sess.graph)
                    train_writer.add_summary(summary, cnt)
                    train_writer.close()
                cnt += 1

                # To print P(first state, first action) of each trajectory
                # first_state_key = str(traj[0]) + ", a: " + str(D_actions[t_i][0])
                # print("\t P({}, a: {}): {:.4f}".format(traj[0], A_s[t_i][s_i], p_vals[first_state_key][-1]))
            ep_time += (time.time()-ep_start)*1000
            if verbose: print("Epoch time: ", (time.time()-ep_start)*1000)
        tot_time += (time.time()-tot_start)*1000
    return p_vals, w, sft_time / sft_ticks, ep_time / n_epochs, tot_time
