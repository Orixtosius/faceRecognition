import numpy as np
import pandas as pd

class dataManipulation():
    
    def __init__(self, path, shuffle, targetIndex, split_ratio) -> None:
        self.path = path
        self.shuffle = shuffle
        self.targetIndex = targetIndex
        self.split_ratio = split_ratio
        self.df = None


    def import_data(self):
        try:
            self.df = pd.read_csv(self.path)
        except:
            raise

    def preprocess_data(self):

        mu = self.X_train.mean(axis = 0)
        std = self.X_train.std(axis = 0)
        self.X_train = (self.X_train - mu) / std
        self.X_test = (self.X_test - mu) / std



    def separate_target(self, seed):

        try:
            data = self.df.to_numpy().astype(np.float32)
        except ValueError:
            if len(self.df.columns) < 3:
                non_target_col = list(self.df.columns).pop(self.targetIndex-1)
                target_col = list(self.df.columns).pop(self.targetIndex)
                df_nonTarget = self.df[non_target_col].str.split(" ", expand=True)
                df_target = pd.DataFrame(self.df[target_col])        
                df_temp = pd.concat([df_target, df_nonTarget], axis = 1)
                data = df_temp.to_numpy().astype(np.float32)
                                
                
        print(f"Shape of data is {data.shape}\nLenght of data is {len(data)}")
        if self.shuffle:
            np.random.seed(seed)
            np.random.shuffle(data)
            
        target = data[:,self.targetIndex]
        train_data = np.delete(data, self.targetIndex, 1)

        M = round(self.split_ratio*len(target))
        print(f"Split will occur after index of {M}...")

        self.Y_train = target[:M]
        self.Y_test = target[M:]

        self.X_train = train_data[:M]
        self.X_test = train_data[M:]

    def get_processed_data(self, seed):
        self.import_data()
        self.separate_target(seed)
        self.preprocess_data()

