from collections import defaultdict

# IRL Interface
import sys
sys.path.append("../")
from IRLProblem import IRLProblem

# GeoLife Wrapper
sys.path.append("../../dataset/GeolifeTrajectories1.3/")
from geolife_data import GeoLifeData

def group_by_goals(trajectories):
    
    goal_to_traj = defaultdict(lambda: [])
    
    for tau in trajectories:
        goal = tau[-1][0]
        goal_to_traj[goal].append(tau)
    return goal_to_traj

class IRL_GeoLifeWorld(IRLProblem):
    
    # TODO: Cache Trajectories and Dyanmics?
    def __init__(self, precache_state_feature=False, *args, **kwargs):
        
        self.geolife = GeoLifeData(*args, **kwargs)
        if precache_state_feature:
            self.geolife.precache_state_feature()
            
        self.phi = self.geolife.phi
        self.trajectories = None
        self.T = None
        
    def features(self, state, img_mode="L", mu=0., std=1.):
        
        return (self.phi(state, img_mode)/255. - mu) / std
    
    def sample_trajectories(self):
        
        if self.trajectories is None:
            
            tau_s_list, tau_mdp_s_list, tau_a_list, traj_dfs = self.geolife.get_trajectories()
            self.trajectories = []
            # Ignore tau_s_list as it's simply tuple representation of tau_mdp_s_list
            for traj_idx, traj_states in enumerate(tau_mdp_s_list):
                traj_actions = tau_a_list[traj_idx]
                self.trajectories.append(list(zip(traj_states, traj_actions)))
        return self.trajectories
    
    def get_dynamics(self):
        if self.T is None:
            self.T = self.geolife.get_dynamics()
        return self.T
