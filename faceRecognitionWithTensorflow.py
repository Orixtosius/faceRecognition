import numpy as np
import pandas as pd
import tensorflow as tf
from get_data import dataManipulation
from train_model import Model

path = "train.csv"

data = dataManipulation( path, 1, 1, 0.7 )
data.get_processed_data()

print(f"*********************************************************")
print(f"\nData has been imported and separated.\n")
print(f"*********************************************************")

multi_class_model = Model(data, batch_size = 1000, l_rate = 0.002, n_iter = 50, reg = 0.0, decay = 0.99, momentum = 0.95)

multi_class_model.linearModel()

