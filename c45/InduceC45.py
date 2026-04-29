import pandas as pd
import json
from c45 import C45Tree

def main(csv_path:str, save_file_path = None):
    x, y, a = read_csv(csv_path)
    new_tree = C45Tree(splitting_metric="ig")
    new_tree.fit(x, y, a, thresh=0.05)
    print(json.dumps(new_tree.tree, indent=2))


def read_csv(csv_path:str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    # read entire csv
    df = pd.read_csv(csv_path)
    #parse new df to get types and class var
    col_names = pd.Series(df.columns)
    data_types = df.iloc[0]
    class_var = df.iloc[1, 0]
    df = df.drop([0, 1])
    col_names= col_names[0: -1]
    data_types = data_types.replace({"0":"float64", r"^[1-9]$":"category"}, regex=True)
    col_types = pd.Series(data_types, index=col_names).to_dict()
    df = df.astype(col_types).reset_index(drop=True)
    #separate to y and X
    y = df.loc[:, class_var]
    x = df.drop(columns=class_var)
    return x, y, pd.Series(col_names)

if __name__ == "__main__":
    main("yellow-small+adult-stretch.csv")