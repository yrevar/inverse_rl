import math
import numpy as np
from mdptoolbox_wrapper import *
import mdptoolbox

class MDPGridWorld(object):
    """
    This is a nice approximation for all the complexity of the entire universe 
        –Charles Isbell
    """
    MARKER_OBSTACLE = '#'
    MARKER_CURR_STATE = '@'
    MARKER_FREE_SPACE = ' '
    TERMINAL_STATE = (-1, -1)
    BIG_NUM = 1e10
    
    def __init__(self, grid, living_reward=-0.01, gamma=0.99, 
                     action_noise_dist=[0.1, 0.8, 0.1], silent=True):
        
        """
        Returns a grid world with input specifications. 
        
        Grid spec: '#' walls, '<int>' terminal state, 'S' initial position, ' ' free space
        """
        assert(len(grid) != 0 and len(grid[0]) != 0)
        
        # Auxiliary
        self.grid = grid.copy()
        self.width = len(grid[0])
        self.height = len(grid)
        
        # States
        self.curr_state_idx, self.states, self.is_obstacle, self.is_terminal, self.r_max, self.r_min \
                                                    = self._parse_grid(grid)
        self.idx_to_state = {i: s for i, s in enumerate(self.states)}
        self.state_to_idx = dict((v, k) for k, v in self.idx_to_state.items())
        self.nS = len(self.states)
        self.state_idxs = list(range(self.nS))
        
        # Absorbing state:
        """
        Value iteration requires approximate value of the next state which is undefined for the terminal states, 
        so we add dummy state with 0 reward and only self loop to itself.
        Absorbing state will prevent accumulating -/+ infinite reward from terminal states
        """
        self.absorbing_state_idx = self.state_to_idx[(self.TERMINAL_STATE)]
        
        # Actions
        self.action_noise_dist = action_noise_dist
        self.actions_name = ["North", "East", "South", "West"]
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.nA = len(self.actions)
        self.action_idxs = range(self.nA)
        # Add class attribute for action strings which evaluates to action index
        for i, a_name in enumerate(self.actions_name):
            setattr(type(self), a_name, i)
        
        # Reward
        self.living_reward = living_reward
        self.R = np.asarray([self.reward(s_idx) for s_idx in range(self.nS)])
        
        # Discounting factor
        self.gamma = gamma
        
        # Transition probability T(S, A, S')
        self.T = self.compute_transition_probability_matrix(self.nS, self.nA, 
                                                            self.action_noise_dist, self.simple_r_n_dynamics) 
        
        # For visualization purpose
        self.silent = silent
        if not self.silent: print(self.get_state_representation_2d(self.curr_state_idx))
    
    def __is_terminal_cell(self, grid, row, col):
        return isinstance(grid[row][col], int) or isinstance(grid[row][col], float)
    
    def __is_terminal_state(self, grid, s_idx):
        row, col = self.idx_to_state[s_idx]
        return isinstance(grid[row][col], int) or isinstance(grid[row][col], float)
    
    def _parse_grid(self, grid):
        """
        Returns: Computes number of states in the grid world.
        Input: 
            grid: grid description as python 2d array of strings (Ref: book_grid)
        """
        states = []
        start_state_idx = 0
        r_max, r_min = float('-inf'), float('inf')
        is_terminal = {}
        is_obstacle = {}
        
        s_idx = 0
        for row in range(self.height):
            for col in range(self.width): # all cols
                
                # Add state
                state = (row, col)
                states.append(state)
                
                
                # Check start state
                if grid[row][col] == self.MARKER_CURR_STATE:
                    start_state_idx = s_idx
                    
                # Mark obstacles
                is_obstacle[s_idx] = grid[row][col] == self.MARKER_OBSTACLE
                
                # Mark terminal states, compute r_max & r_min values
                if self.__is_terminal_cell(self.grid, row, col):
                    r_max = max(self.grid[row][col], r_max)
                    r_min = min(self.grid[row][col], r_min)
                    is_terminal[s_idx] = True
                else:
                    is_terminal[s_idx] = False
                
                s_idx += 1
                
        # +absorbing state
        states.append(self.TERMINAL_STATE)
        
        return start_state_idx, states, is_obstacle, is_terminal, r_max, r_min
    
    def reward(self, s_idx):
        """
        Reward function, f: state -> reward
        Returns reward of the given state referred using its index
        """
        
        if s_idx == self.absorbing_state_idx:
            return 0
        r, c = self.idx_to_state[s_idx]
        if self.__is_terminal_cell(self.grid, r, c):
            return float(self.grid[r][c])
        elif self.grid[r][c] == self.MARKER_OBSTACLE:
            return -self.BIG_NUM
        else:
            return self.living_reward
        
    def sample_next_state(self, s_idx, a_idx):
        """
        Returns: the next state (index) from executing stochastic a_idx in s_idx
        """
        return np.random.choice(self.state_idxs, p=self.T[s_idx, a_idx])
        
    def compute_transition_probability_matrix(self, nS, nA, a_noise_dist, dynamics):
        
        T = np.zeros((nS, nA, nS))
        for s_idx in self.state_idxs: # no transitions from absorbing state
            for a_idx in range(nA):
                
                # For any action in terminal or absorbing state, we transition to absorbing state
                if self.__is_terminal_state(self.grid, s_idx) or s_idx == self.absorbing_state_idx:
                    T[s_idx, a_idx, self.absorbing_state_idx] = 1.
                else:
                    for k, (s_prime_idx, next_p) in enumerate(dynamics(s_idx, a_idx, a_noise_dist)):
                        T[s_idx, a_idx][s_prime_idx] = next_p
        return T
    
    def simple_r_n_dynamics(self, s_idx, a_idx, a_noise_dist):
        """
        Implements dynamics as specified for toy mdp gridworld in R&N chap. 17
        Returns: Returns transition probabilities for taking stochastic action a_idx in state s_idx
                    [(s_prime, p(s_prime)), ...]
        """
        # Taking action leads to the state that's located in the direction of action. 
        # Because actions (dynamics) are assumed stochastic, we could also land in undesired states 
        # with some probability as defined in a_noise_dist. 
        # This dynamics assumes that taking action could lead to either the desired cell or any of the two cells on sideways
        # E.g., Taking action North could lead to either North or East or West cells with probability
        # specified as [P[West], P[North], P[East]] in a_noise_dist
        # This  allows simple cyclic retrieval of next states with action sorted as ["North", "East", "South", "West"]
        assert(len(a_noise_dist) == 3) 
        
        next_state_to_prob = {}
        r, c = self.idx_to_state[s_idx]
        
        for j, noisy_a_idx in enumerate([a_idx-1, a_idx, (a_idx+1)%self.nA]): # hard coded for len(a_noise_dist) == 3
            
            dr, dc = self.actions[noisy_a_idx]
            p = a_noise_dist[j]
            new_r, new_c = r+dr, c+dc
            if new_r < 0 or new_r >= self.height \
                or new_c < 0 or new_c >= self.width \
                or self.grid[new_r][new_c] == self.MARKER_OBSTACLE:
                    new_r, new_c = r, c
                    
            if self.state_to_idx[(new_r, new_c)] in next_state_to_prob:
                next_state_to_prob[self.state_to_idx[(new_r, new_c)]] += p
            else:
                next_state_to_prob[self.state_to_idx[(new_r, new_c)]] = p
            
        return next_state_to_prob.items()
    
    def _get_optimal_policy(self):
        """
        Returns: Optimal Policy Pi(s) for all s
        """
        if 'vi_results' not in dir(self):
            self.vi_results = run_value_iteration(self.T, self.R, self.gamma)
            
        return np.asarray(self.vi_results.policy).copy()
    
    def get_state_feature(self, s_idx):
        return np.eye((self.nS))[s_idx].tolist()
    
    def get_state_2d_features_matrix(self):
        """
        Returns: State feature matrix of size N x K, N is number of states and K is dimensionality of state features
        """
        return np.asarray(self.states).copy()
    
    def get_state_grid_features_matrix(self):
        """
        Returns: State feature matrix of size N x K, N is number of states and K is dimensionality of state features
        """
        return np.asarray([self.get_state_feature(s) for s in self.state_idxs])
    
    def get_valid_random_state(self):
        
        s_idx = np.random.randint(self.nS-1)
        r, c = self.states[s_idx]
        while self.grid[r][c] == self.MARKER_WALL:
            return self.get_valid_random_state()
        return s_idx
    
    ## Trajectory sampling functions
    def get_random_state(self, avoid_obtacles=False):
        
        s_idx = np.random.randint(self.nS-1) # ignore absorbing state
        if avoid_obtacles:
            r, c = self.states[s_idx]
            while self.grid[r][c] == self.MARKER_OBSTACLE:
                return self.get_random_state(avoid_obtacles=True)
        return s_idx
    
    def sample_trajectories(self, M, init_state_idx="random", max_length=100, Pi="optimal", padding=True):
        """
        Samples and returns T trajectories of the form 
            [ [(s0_0, a0_0, r0_0), ..., (s0_T, a0_T, r0_T)], [(s1_0, a1_0, r1_0), ..., (s1_T, a1_T, r1_T)], 
                                                ... [(sM_0, aM_0, rM_0), ..., (sM_T, aM_T, rM_T)]]
        Input:
            M: # of trajectories to be sampled
            init_state_idx: initial state s0, if assigned "random" it'll randomly sample starting state
            max_length: maximum length of the trajectory
            padding: If trajectory (walk over DAG) terminates before encountering max_length states, it'll pad the last (s,a,r)
            Pi: Policy. Mapping of state_idx to action_idx of size N, where N is number of states.
            
        """
        tau_list = []
        for _ in range(M):
            tau_list.append(self.sample_trajectory(init_state_idx, max_length, Pi, padding))
        return tau_list
    
    def sample_trajectory(self, init_state_idx="random", max_length=100, Pi="optimal", behavior_optimality=1., padding=True):
        """
        Samples a trajectory of the form [(s0, a0, r0), ..., (sT, aT, rT)]
        Input:
            init_state_idx: initial state s0, if assigned "random" it'll randomly sample starting state
            max_length: maximum length of the trajectory
            padding: If trajectory (walk over DAG) terminates before encountering max_length states, it'll pad the last (s,a,r)
            Pi: Policy. Mapping of state_idx to action_idx of size N, where N is number of states.
                Two special values:
                    - "random" will initialize a random policy
                    - "optimal" will use the optimal policy for the given MDP (currently using policy iteration)
        """
        tau = []
        absorbing_state_idx = self.absorbing_state_idx
        
        # Initial state
        s_idx = self.get_random_state(avoid_obtacles=True) if init_state_idx == "random" else  init_state_idx

        # Initial policy
        if Pi == "random":    
            Pi = np.random.choice(self.nA, size=self.nS-1)
        elif Pi == "optimal":
            Pi = self._get_optimal_policy()
        else:
            pass
        
        for i in range(max_length):
            
            # Additional noise in trajectory, needed?
            if behavior_optimality < 1:
                if np.random.random() <= behavior_optimality:
                    a_idx = Pi[s_idx]
                else:
                    a_idx = np.random.choice(self.nA)
            else:
                a_idx = Pi[s_idx]
                
            s_prime_idx = self.sample_next_state(s_idx, a_idx)
            tau.append((s_idx, a_idx, self.R[s_prime_idx]))
            if s_idx == self.absorbing_state_idx: #self._is_terminal(s_idx):
                # Reached terminal state, terminate trajectory here
                break
            s_idx = s_prime_idx
        
        if padding:
            for j in range(i, max_length):
                tau.append((s_idx, a_idx, self.R[s_prime_idx]))
        return tau
    
    def interpret_trajectory(self, tau):
        
        for (s_idx, a_idx, r) in tau:
            if s_idx == self.absorbing_state_idx:
                break
            print("s:")
            print(self.get_state_representation_2d(s_idx))
            print("a: ", self.actions_name[a_idx])
            print("r: ", r) # Note: this is current state reward
    
    ## Printing functions
    def disp_custom_grid(self, state_values, formatting=lambda x: "{:+.3f}".format(x), display_absorbing=True):
        
        """
        Returns: state_values representation of states
        Input: state_values associated with each state
        """
        self.state_values_dict = {self.states[i]: state_values[i] for i in range(len(state_values))}
        
        msg = ''
        cell_filler = "_"
        grid = self.grid
        for r in range(self.height):
            for c in range(self.width):
                #if grid[r][c] != self.MARKER_OBSTACLE:
                tt = formatting(self.state_values_dict[(r,c)])
                #else: # Show unreachable states (such as Walls) with -inf
                #    tt = str(np.float("-inf"))
                msg += tt + "\t" #"{txt:{fill}^5s}".format(txt=tt, fill=cell_filler)
            msg += "\n"
        
        if display_absorbing and len(state_values)-1 == self.absorbing_state_idx:
            msg += "Absorbing state: " + formatting(state_values[-1])
        print(msg)
        
    def get_state_representation_2d(self, s_idx):
        
        msg, cell_filler = '', "_"
        grid = self.grid
        curr_r, curr_c = self.idx_to_state[s_idx]
        for r in range(self.height):
            for c in range(self.width):
                if r == curr_r and c == curr_c:
                    tt = self.MARKER_CURR_STATE
                elif grid[r][c] == self.MARKER_FREE_SPACE\
                    or (grid[r][c] == self.MARKER_CURR_STATE and (r != curr_r or c != curr_c)):
                    tt = cell_filler
                else:
                    if isinstance(grid[r][c], int) or isinstance(grid[r][c], float):
                        tt = "{:+d}".format(grid[r][c])
                    else:
                        tt = grid[r][c]
                msg += "{txt:{fill}^5s}\t".format(txt=tt, fill=cell_filler)
            msg += "\n"
        return msg
    def __str__(self):
        return self.get_state_representation_2d(self.curr_state_idx)

# Mdptoolbox utils
def run_value_iteration(T, R, gamma, *args, **kwarg):
    T = T.transpose(1,0,2)
    vi = mdptoolbox.mdp.ValueIteration(T, R, gamma, *args, **kwarg)
    vi.run()
    return vi