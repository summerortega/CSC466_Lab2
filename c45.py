import pandas as pd


class C45:
    def __init__(self, splitting_metric = 'igr', splitting_threshold = '0.05'):
        self.splitting_metric = splitting_metric
        self.splitting_threshold = splitting_threshold
        self.tree = {}
        pass
    # give params default vals
    # add optional input params with default vals
    # attrbutes is a
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
                            {"var": att.index,
                             "edges": []
                            }
                        }
            for val in att:
                x_filtered = x[x[att == val]]
                y_filtered =  y[y[att == val]]
                #how to attach dictionary value
                new_tree = self.fit(x_filtered, y_filtered, a.drop(att), thresh)

                att_tree["node"]["edges"].append({"val": val } | new_tree )

            curr_tree["node"] = att_tree
            return self.tree

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


