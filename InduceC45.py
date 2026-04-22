# create instance of c45 class
# use training data to make tree
# output prints out the description of the induced
# decision tree in JSON format.
# Essentially, you can view the InduceC45.py program as a wrapper around the .fit() and .save_tree()
# methods of the c45 class
from c45 import C45
import pandas as pd

C45()
def read_csv(test_csv):
    # read csv
    df = pd.read_csv(test_csv)

    pass