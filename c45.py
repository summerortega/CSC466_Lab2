import pandas as pd
import numpy as np
import json

class C45Tree:
    def __init__(self, splitting_metric = 'igr', splitting_threshold = '0.05'):
        self.splitting_metric = splitting_metric
        self.splitting_threshold = splitting_threshold
        self.tree = {}


    def fit(self, x:pd.DataFrame, y:pd.Series, a:pd.DataFrame|pd.Series, thresh:float):
        curr_tree = {}
        #Base Case 1: No Attributes Left to Split on
        if a.shape[0] == 0:
            decision = y.mode().values[0]
            p = len(y[y == decision]) / len(y)
            return {"leaf":
                        {"decision": decision,
                         "probability":p}
                  }
        #Base Case 2: y (class attribute set) is homogenous
        elif y.value_counts().iloc[0] == y.shape[0]:
            decision = y.mode().values[0]
            p = len(y[y == decision]) / len(y)
            return {"leaf":
                        {"decision": decision,
                         "probability": p}
                    }
        else:
            att, split_val = select_split_att(x, y, a, thresh, mode=self.splitting_metric)
            #Case 1: No attribute returned; Return a leaf
            if not att:
                decision = y.mode().values[0]
                p = len(y[y == decision]) / len(y)
                return {"leaf":
                            {"decision": decision,
                             "probability": p}
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
                        if x_filtered.shape[0] != 0:
                            new_tree = self.fit(x_filtered, y_filtered, a[a != att].reset_index(drop=True), thresh)
                            att_tree["node"]["edges"].append({"edge": {"value": val } | new_tree })
                        else:
                            decision = y.mode().values[0]
                            p = len(y[y == decision]) / len(y)
                            return {"leaf":
                                        {"decision": decision,
                                         "probability": p}
                                    }
                else:
                    # numeric
                    x_filtered_lt = x[x[att] <= split_val]
                    y_filtered_lt = y[x[att] <= split_val]
                    x_filtered_gt = x[x[att] > split_val]
                    y_filtered_gt = y[x[att] > split_val]

                    #left child
                    new_tree = self.fit(x_filtered_lt, y_filtered_lt, a[a != att].reset_index(drop=True), thresh)
                    att_tree["node"]["edges"].append({"edge": {"value": split_val,
                                                               "op": '<='}} | new_tree)

                    #right child
                    new_tree = self.fit(x_filtered_gt, y_filtered_gt, a[a != att].reset_index(drop=True), thresh)
                    att_tree["node"]["edges"].append({"edge": {"value": split_val,
                                                      "op": '>'}} | new_tree )

                curr_tree = att_tree
        self.tree = curr_tree
        return curr_tree


    def get_prediction(self, x:pd.Series) -> str:
        decision = None
        curr_node = self.tree["node"]
        while not decision:
            curr_var = curr_node["var"]
            edges =  {edge["edge"]["val"]: edge["edge"] for edge in curr_node["edges"]}
            edge = edges[x[curr_var]]
            node_keys = list(edge.keys())
            if node_keys[1] == "node":
                curr_node = edge["node"]
            elif node_keys[1] == "leaf":
                decision = edge["leaf"]["decision"]
        return decision


    def predict(self, x_test:pd.DataFrame) -> list[str]:
        results = [self.get_prediction(x_test.iloc[i]) for i in range(len(x_test))]
        return results


    def save_tree(self, save_file_path):
        # saves the tree to a file
        # creates/overwrited files, places it into JSON rendering of the tree
        with open(save_file_path, "w") as f:
            json.dump(self.tree, f, indent=2)


    def read_tree(self, load_file_path):
        # reads the tree from a file
        # reads the JSON rendering of the tree, sets value = self.tree
        with open(load_file_path, "r") as f:
            self.tree = json.loads(f.read())


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
            value, gain = find_best_split(x, y, att, mode)
            numeric_splits[att] = value
            metric.append(gain)
    best = np.argmax(metric)
    if metric[best] >= thresh:
        if a[best] in numeric_splits.keys():
            return a[best], numeric_splits[a[best]]
        else:
            return a[best], None
    else:
        return None, None


def find_best_split(x, y, att, mode):
    unique = np.unique(x[att])[1:]
    results = []
    for val in unique:
        x_filtered_lt = x[x[att] <= val]
        y_filtered_lt = y[x[att] <= val]
        x_filtered_gt = x[x[att] > val]
        y_filtered_gt = y[x[att] > val]

        entropy_binary_split = ((x_filtered_lt.shape[0]/x.shape[0] * entropy(y_filtered_lt)) +
                  (x_filtered_gt.shape[0]/x.shape[0] * entropy(y_filtered_gt)))
        if mode == "ig":
            results.append(entropy(y) - entropy_binary_split)
        elif mode == "igr":
            gain = entropy(y) - entropy_binary_split
            results.append(info_gain_ratio_numeric(gain, x, y, att))

    best = np.argmax(results)
    return unique[best], results[best]


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
    filtered_y_shapes = [y[x[att] == val].shape[0] for val in values]
    y_shape = y.shape[0]
    den = [(filtered_y/y_shape) * (np.log2((filtered_y/y_shape))) for filtered_y in filtered_y_shapes]
    return gain / (-1 * sum(den))


def info_gain_ratio_numeric(gain, x, y, att):
    unique = np.unique_counts(x[att])
    values = unique.values
    filtered_y_shapes = [y[x[att] == val].shape[0] for val in values]
    y_shape = y.shape[0]
    den = [(filtered_y / y_shape) * (np.log2((filtered_y / y_shape))) for filtered_y in filtered_y_shapes]
    return gain / (-1 * sum(den))


def entropy (y):
    if y.value_counts().iloc[0] == y.shape[0]:
        return 0
    unique = np.unique_counts(y)
    counts = unique.counts
    total = np.sum(counts)
    calculations = [(counts[i] / total) * np.log2((counts[i] / total)) for i in range(len(counts))]

    return -1 * np.sum(calculations)



