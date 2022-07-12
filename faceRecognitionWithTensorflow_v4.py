from get_data import dataManipulation
from train_model_v3 import Model
import time

path = "train.csv"

data = dataManipulation( path, 1, 0, 0.7 )
data.get_processed_data(seed = 100)

print(f"*********************************************************")
print(f"\nData has been imported and separated.\n")
print(f"*********************************************************")

multi_class_model = Model(data)
start_time = time.time()
multi_class_model.linearModel(batch_size = 250, l_rate = 0.0002, n_iter = 20, reg = 0.0, decay = 0.99, momentum = 0.95)
break_time_1 = time.time()
print(f"Time passed for first model is {break_time_1 - start_time}")

start_time_2 = time.time()
multi_class_model.addLayer(128)
multi_class_model.linearModel(batch_size = 100, l_rate = 0.0001, n_iter = 50, reg = 0.2, decay = 0.99, momentum = 0.95)
break_time_2 = time.time()
print(f"Time passed for second model is {break_time_2 - start_time_2}")

start_time_3 = time.time()
multi_class_model.addLayer(128)
multi_class_model.addLayer(64)
multi_class_model.linearModel(batch_size = 250, l_rate = 0.0002, n_iter = 50, reg = 0.5, decay = 0.98, momentum = 0.99)
break_time_3 = time.time()
print(f"Time passed for third model is {break_time_2 - start_time_3}")

print("Comparation of effect of hidden layers on models\n")
for history in multi_class_model.pastModels:
    print(f"Model {history['model_id']}\nLayers : {history['layers']} - Batch Size {history['n_batch']}")
    print(f"Model Iteration {history['n_iter']} - Learning Rate {history['l_rate']} - Regulation {history['reg']}")
    print(f"Test Error {history['test_error']}")
    print("------------------------------------------------------------------------------------------------------\n")

