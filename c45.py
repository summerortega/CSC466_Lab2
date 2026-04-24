import pandas as pd
import numpy as np

class C45:
    def __init__(self, splitting_metric = 'igr', splitting_threshold = '0.05'):
        self.splitting_metric = splitting_metric
        self.splitting_threshold = splitting_threshold
        self.tree = {}
        pass
    # give params default vals
    # add optional input params with default vals
    # attributes is a
    def fit(self, x:pd.DataFrame, y:pd.Series, a:pd.DataFrame|pd.Series, thresh:float):
        curr_tree = {}
        if x.shape[0] == 0:
            return {"leaf":
                        {"decision": y.mode().values[0],
                         "probability":0}
                    }
        elif y.value_counts()[0] == y.shape[0]:
            return {"leaf":
                        {"decision": y.mode().values[0],
                         "probability": 0}
                    }
        else:
            att = select_split_att(x, y, a, thresh)
            if len(att) == 0:
                return {"leaf":
                            {"decision": y.mode().values[0],
                             "probability": 0}
                        }
            att_tree = {"node":
                            {"var": att.index[0],
                             "edges": []
                            }
                        }
            for val in att:
                x_filtered = x[x[att == val]]
                y_filtered =  y[y[att == val]]
                #how to attach dictionary value
                new_tree = self.fit(x_filtered, y_filtered, a.drop(att), thresh)

                if x[att].dtypes == 'category' :
                    att_tree["node"]["edges"].append({"edge": {"val": val }} | new_tree )
                elif x[att].dtypes == 'float64' :
                    att_tree["node"]["edges"].append({"edge": {"val": val,
                                                      "op": op}} | new_tree )
                    # TODO: figure out how to determine op (greater than or less than)

            curr_tree = att_tree
        return curr_tree

        # build the tree
        # returns an object representing the best tree
        # c45 algo goes here


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
def select_split_att(x, y, a, thresh, mode):
    metric = []
    for att in a:
        if mode == 1:
            metric.append(infoGain(x, y, att))
        elif mode == 0:
            metric.append(infoGainRatio(x, y, att))

    best = np.argmax(metric)
    if metric[best] >= thresh:
        return best
    else:
        return None

def infoGain(x, y, att):
    unique = np.unique_counts(x[att])
    counts = unique.counts
    values = unique.values
    djs = [y[x[att] == val] for val in values]
    return np.sum([(dj.shape[0]/y.shape[0]) * entropy(dj) for dj in djs])

def infoGainRatio(x, y, att):
    gain = infoGain(x, y, att)
    unique = np.unique_counts(x[att])
    values = unique.values
    djs = [y[x[att] == val].shape[0] for val in values]
    y_shape = y.shape[0]
    den = [(dj/y_shape) * (np.log2((dj/y_shape))) for dj in djs]
    return gain / (-1 * sum(den))

def entropy (y):
    if y.value_counts()[0] == y.shape[0]:
        return 0
    unique = np.unique_counts(y)
    counts = unique.counts
    total = counts.sum
    calculations = [(counts[i] / total) * np.log2((counts[i] / total)) for i in range(len(counts))]

    return -1 * sum(calculations)


