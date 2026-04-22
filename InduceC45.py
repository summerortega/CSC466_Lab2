# create instance of c45 class
# use training data to make tree
# output prints out the description of the induced
# decision tree in JSON format.
# Essentially, you can view the InduceC45.py program as a wrapper around the .fit() and .save_tree()
# methods of the c45 class
from scrapy.spiders.sitemap import regex

from c45 import C45
import pandas as pd

tree = C45()
def read_csv(test_csv):
    # read entire csv
    df = pd.read_csv(test_csv)
    #parse new df to get types and class var
    col_names = df.columns
    data_types = df.iloc[0]
    class_var = df.iloc[1][0]
    df = df.drop([0, 1])
    data_types = data_types.replace({"0":"float64", r"^[1-9]$":"category"}, regex=True)
    col_types = pd.Series(data_types, index=col_names).to_dict()
    df = df.astype(col_types).reset_index(drop=True)
    #separate to y and X
    y = df.loc[:, class_var]
    X = df.drop(columns=class_var)


    pass