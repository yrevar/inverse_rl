import os
import sys
# try:
#     import cPickle as pickle
# except ModuleNotFoundError:
# import pickle
import dill as pickle

class PickleWrapper(object):

    def __init__(self, store_file="./exp_data.p"):

        self.results_file = store_file
        self.results_dir = os.path.dirname(self.results_file)

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        self.load()

    def dump(self, obj):
        with open(self.results_file, 'wb') as fp:
            pickle.dump(obj, fp)

    def load(self):
        if os.path.exists(self.results_file):
            with open(self.results_file, 'rb') as fp:
                self.obj = pickle.load(fp)
        else:
            self.obj = {}
        return self.obj

class ExperimentHelper(object):
    
    def __init__(self, config_dict, data_file="./exp1/exp_data.p"):
        
        self.config_dict = config_dict
        self.data_file = data_file
        if data_file:
            self.data_dir = os.path.dirname(data_file)
            os.makedirs(self.data_dir, exist_ok=True)
            self.exp_data_storage = PickleWrapper(data_file)
            self.exp_data = self.exp_data_storage.load()
        
        self.dump("config", config_dict)
        
    def dump(self, key, value):
        
        if self.data_file:
            self.exp_data[key] = value
            self.exp_data_storage.dump(self.exp_data)
            
    def load(self, key, default=None):
        
        if self.data_file:
            if key in self.exp_data:
                return self.exp_data[key]
            else:
                return default
        else:
            raise ValueError("No data file configured!")
            
            
    def launch(self, here=globals()):
        here.update(self.config_dict)
        #         thismodule = sys.modules[__name__]
        #         for k,v in self.config_dict.items():
        #             setattr(thismodule, k, v)
        