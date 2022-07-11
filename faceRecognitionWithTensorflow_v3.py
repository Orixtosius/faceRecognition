from get_data import dataManipulation
from train_model_v2 import Model

path = "train.csv"

data = dataManipulation( path, 1, 0, 0.7 )
data.get_processed_data(seed = 100)

print(f"*********************************************************")
print(f"\nData has been imported and separated.\n")
print(f"*********************************************************")

multi_class_model = Model(data)

multi_class_model.linearModel(batch_size = 1000, l_rate = 0.0002, n_iter = 20, reg = 0.0, decay = 0.99, momentum = 0.95)

#multi_class_model.addLayer(128)
#multi_class_model.linearModel(batch_size = 100, l_rate = 0.0002, n_iter = 50, reg = 0.0, decay = 0.99, momentum = 0.95)
#
#multi_class_model.addLayer(64)
#multi_class_model.linearModel(batch_size = 100, l_rate = 0.0002, n_iter = 50, reg = 0.0, decay = 0.99, momentum = 0.95)


print("Comparation of effect of hidden layers on models\n")
for history in multi_class_model.pastModels:
    print(f"Model {history['model_id']} ===>  Layers : {history['layers']} and Test Error {history['test_error']}")
    print("------------------------------------------------------------------------------------------------------\n")

