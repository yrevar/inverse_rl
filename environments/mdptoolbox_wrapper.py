import json
import numpy as np

class MDPToolBoxWrapper:
    """Convert the plain object description of the mdp into gamma and T and R matrices"""
    def __init__(self, descr):
        self.descr = descr
        self.gamma = descr["gamma"]
        self.nS = len(descr["states"])
        self.nA = len(descr["states"][0]["actions"])
        self.transitions = np.zeros((self.nA, self.nS, self.nS))
        self.rewards = np.zeros((self.nA, self.nS, self.nS))
        state_indexes = {state["id"]: i for i, state in enumerate(descr["states"])}
        for state in descr["states"]:
            assert len(state["actions"]) == self.nA, "All states must have same number of possible actions"
            for i, action in enumerate(state["actions"]):
                for transition in action["transitions"]:
                    state_index = state_indexes[state["id"]]
                    new_state_index = state_indexes[transition["to"]]
                    self.transitions[i, state_index, new_state_index] = transition["probability"]
                    self.rewards[i, state_index, new_state_index] = transition["reward"]
                    
def create_MDP(P, gamma):
    
    mdp = {}
    mdp["gamma"] = gamma
    mdp["states"] = []
        
    for s in P:
        mdp["states"].append({})
        mdp["states"][-1]["id"] = s
        mdp["states"][-1]["actions"] = []
        
        for a in P[s]:
            mdp["states"][-1]["actions"].append({})
            mdp["states"][-1]["actions"][-1]["id"] = a
            mdp["states"][-1]["actions"][-1]["transitions"] = []
            
            for i, (prob, s_prime, reward, done) in enumerate(P[s][a]):
                
                mdp["states"][-1]["actions"][-1]["transitions"].append({})
                mdp["states"][-1]["actions"][-1]["transitions"][-1]["id"] = i
                mdp["states"][-1]["actions"][-1]["transitions"][-1]["probability"] = prob
                mdp["states"][-1]["actions"][-1]["transitions"][-1]["reward"] = reward
                mdp["states"][-1]["actions"][-1]["transitions"][-1]["to"] = s_prime
    return mdp

def create_json_from_P(P, gamma, name, write=True):
    
    mdp = create_MDP(P, gamma)
    js = json.dumps(mdp, ensure_ascii=False, sort_keys=True, indent=2)
    if write:
        with open("./" + name + ".json", "w") as text_file:
            text_file.write(js)
    return mdp