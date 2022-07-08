import numpy as np
import pandas as pd
import tensorflow as tf
from get_data import dataManipulation

path = "train.csv"

data = dataManipulation( path, 1, 1, 0.7 )
data.get_processed_data()

print(f"Sample data from X_train: \n{ data.X_train }")
