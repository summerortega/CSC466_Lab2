class C45:
    def __init__(self, splitting_metric = 'igr', splitting_threshold = '0.05'):
        self.splitting_metric = splitting_metric
        self.splitting_threshold = splitting_threshold
        self.tree = {}
        pass
    # give params default vals
    # add optional input params with default vals

def fit(self, training_set, training_set_ground_truth):
    # build the tree
    # returns an object representing the best tree
    # c45 algo goes here
    pass

def predict(self, x_test):
    # takes array of data points we want predictions made for
    # returns array of predictions
    pass

def save_tree(self, filename):
    # saves the tree to a file
    # creates/overwrited files, places it into JSON rendering of the tree
    pass

def read_tree(self, filename):
    # reads the tree from a file
    # reads the JSON rendering of the tree, sets value = self.tree
    pass


