import os
# try:
#     import cPickle as pickle
# except ModuleNotFoundError:
#     import pickle
import dill as pickle

class PickleWrapper(object):

    def __init__(self, store_file="./results.p"):

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
            self.obj = {} #defaultdict(lambda: defaultdict(dict)) is best but not serializable through pickle
        return self.obj
