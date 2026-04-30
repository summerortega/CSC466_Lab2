import pandas as pd
import numpy as np
import json

class C45Tree:
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
        #Base Case 1: No Attributes Left to Split on
        if a.shape[0] == 0:
            return {"leaf":
                        {"decision": y.mode().values[0],
                         "probability":0}
                  }
        #Base Case 2: y (class attribute set) is homogenous
        elif y.value_counts().iloc[0] == y.shape[0]:
            return {"leaf":
                        {"decision": y.mode().values[0],
                         "probability": 0}
                    }
        else:
            att, split_val = select_split_att(x, y, a, thresh, mode=self.splitting_metric)
            #Case 1: No attribute returned; Return a leaf
            #TODO: compute probabilities
            if not att:
                return {"leaf":
                            {"decision": y.mode().values[0],
                             "probability": 0}
                        }
            else:
                att_tree = {"node":
                                {"var": att,
                                 "edges": []
                                 }
                            }
                if not split_val:
                    # category
                    vals = x[att].unique()
                    for val in vals:
                        x_filtered = x[x[att] == val]
                        y_filtered =  y[x[att] == val]
                        new_tree = self.fit(x_filtered, y_filtered, a[a != att].reset_index(drop=True), thresh)
                        att_tree["node"]["edges"].append({"edge": {"val": val } | new_tree })

                else:
                    # numeric
                    x_filtered_lt = x[x[att] <= split_val]
                    y_filtered_lt = y[x[att] <= split_val]
                    x_filtered_gt = x[x[att] > split_val]
                    y_filtered_gt = y[x[att] > split_val]

                    #left child
                    new_tree = self.fit(x_filtered_lt, y_filtered_lt, a.drop(att), thresh)
                    att_tree["node"]["edges"].append({"edge": {"val": split_val,
                                                               "op": '<='}} | new_tree)

                    #right child
                    new_tree = self.fit(x_filtered_gt, y_filtered_gt, a.drop(att), thresh)
                    att_tree["node"]["edges"].append({"edge": {"val": split_val,
                                                      "op": '>'}} | new_tree )

                curr_tree = att_tree
        self.tree = curr_tree
        return curr_tree

        # build the tree
        # returns an object representing the best tree
        # c45 algo goes here

    def predict(self, x_test):
        # takes array of data points we want predictions made for
        # returns array of predictions
        pass

    def save_tree(self, save_file_path):
        # saves the tree to a file
        # creates/overwrited files, places it into JSON rendering of the tree
        with open(save_file_path, "w") as f:
            json.dump(self.tree, f, indent=2)
        pass

    def read_tree(self, filename):
        # reads the tree from a file
        # reads the JSON rendering of the tree, sets value = self.tree
        pass

def select_split_att(x, y, a, thresh, mode):
    metric = []
    numeric_splits = {}
    for att in a:
        if x[att].dtypes == 'category':
            if mode.lower() == 'ig':
                metric.append(info_gain(x, y, att))
            elif mode.lower() == 'igr':
                metric.append(info_gain_ratio(x, y, att))
        elif x[att].dtypes == 'float64':
            value, gain = (find_best_split(x, y, att, thresh))
            #TODO info gain ration for numeric
            numeric_splits[att] = value
            metric.append(gain)
    best = np.argmax(metric)
    if metric[best] >= thresh:
        if a[best] in numeric_splits.keys():
            return a[best], numeric_splits[a[best]]
        else:
            return a[best], None
    else:
        return None

def find_best_split(x, y, att, thresh):
    unique = np.unique(x[att])[:1]
    gain = []
    for val in unique:
        x_filtered_lt = x[x[att] <= val]
        y_filtered_lt = y[x[att] <= val]
        x_filtered_gt = x[x[att] > val]
        y_filtered_gt = y[x[att] > val]

        entropy_binary_split = ((x_filtered_lt.shape[0]/x.shape[0] * entropy(y_filtered_lt)) +
                  (x_filtered_gt.shape[0]/x.shape[0] * entropy(y_filtered_gt)))
        gain.append(entropy(y) - entropy_binary_split)

    best = np.argmax(gain)
    if gain[best] >= thresh:
        return unique[best], gain[best]
    else:
        return None

def info_gain(x, y, att):
    unique = np.unique_counts(x[att])
    values = unique.values
    filtered_y_sets = [y[x[att] == val] for val in values]
    entropy_split = np.sum([y_prime.shape[0]/y.shape[0] * entropy(y_prime) for y_prime in filtered_y_sets])
    return float(entropy(y) - entropy_split)

def info_gain_ratio(x, y, att):
    gain = info_gain(x, y, att)
    unique = np.unique_counts(x[att])
    values = unique.values
    djs = [y[x[att] == val].shape[0] for val in values]
    y_shape = y.shape[0]
    den = [(dj/y_shape) * (np.log2((dj/y_shape))) for dj in djs]
    return gain / (-1 * sum(den))

def entropy (y):
    if y.value_counts().iloc[0] == y.shape[0]:
        return 0
    unique = np.unique_counts(y)
    counts = unique.counts
    total = np.sum(counts)
    calculations = [(counts[i] / total) * np.log2((counts[i] / total)) for i in range(len(counts))]

    return -1 * np.sum(calculations)



