import pandas as pd
import json
import sys
from c45 import C45Tree

def main(csv_path:str, save_file_path:str = None):
    x, y, a = read_csv(csv_path)
    new_tree = C45Tree(splitting_metric="igr")
    new_tree.fit(x, y, a, thresh=0.05)
    new_tree.tree = {"dataset": csv_path} | new_tree.tree
    if not save_file_path:
        print(json.dumps(new_tree.tree, indent=2))
    else:
        new_tree.save_tree(save_file_path)


def read_csv(csv_path:str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    # read entire csv
    df = pd.read_csv(csv_path)
    #parse new df to get types and class var
    col_names = pd.Series(df.columns)
    data_types = df.iloc[0].astype("int64")
    data_types = data_types.astype("string")
    class_var = df.iloc[1, 0]
    df = df.drop([0, 1])
    a = col_names[0: -1]
    data_types = data_types.replace({"0":"float64", r"^[1-9]$":"category"}, regex=True)
    col_types = pd.Series(data_types, index=col_names).to_dict()
    df = df.astype(col_types).reset_index(drop=True)
    #separate to y and X
    y = df.loc[:, class_var]
    x = df.drop(columns=class_var)
    return x, y, pd.Series(a)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 1:
        main(args[0])
    elif len(args) == 2:
        main(args[0], args[1])
    else:
        print("Usage: python3 predict.py <csv_path> [<save_file_path>]")