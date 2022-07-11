import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class Model():
    def __init__(self, data) -> None:
        
        self.X_train = data.X_train
        self.Y_train = data.Y_train
        self.X_test = data.X_test
        self.Y_test = data.Y_test

        self.layers = []
        self.weights = []
        self.bias = []

        self.pastModels = []
        self.modelNo = 0

        self.test_error = []
        self.final_training_error = -1

    
    def addLayer(self, density):
        self.layers.append(density)

    def indicatorMatrix(self, y):
        N = len(y)
        y = y.astype(np.int32)
        K = y.max() + 1
        ind = np.zeros((N,K))
        for n in range(N):
            k = y[n]
            ind[n,k] = 1
        return ind

    def initializeParameters(self, Y_ind_train):
        
        M, NX = self.X_train.shape
        K = Y_ind_train.shape[1]
        print(f"Other Shape in initilation {Y_ind_train.shape[0]}")
        self.layers.insert(0,NX)
        self.layers.append(K)
        L = self.layers
        
        for i in range( len(L)-1 ):
            
            W = np.random.randn( L[i], L[i+1] ) / np.sqrt( L[i] )
            b = np.zeros( L[i+1] )
            self.weights.append(W)
            self.bias.append(b)

    def error_rate(self, p_y, t):
        return np.mean(p_y != t)

    def linearModel(self, batch_size = 100, l_rate = 0.003, n_iter = 100, reg = 0.0, decay = 0.99, momentum = 0.9 ):

        self.l_rate = l_rate
        self.n_iter = n_iter
        self.reg = reg
        self.decay = decay
        self.momentum = momentum
        self.b_size = batch_size
        Y_train_ind = self.indicatorMatrix(self.Y_train)
        Y_test_ind = self.indicatorMatrix(self.Y_test)

        self.initializeParameters(Y_train_ind)
        n_batch = ceil(self.X_train.shape[0]/self.b_size)

        X = tf.placeholder(tf.float32, shape = (None, self.X_train.shape[1]), name = 'X')
        T = tf.placeholder(tf.float32, shape = (None, Y_train_ind.shape[1]), name = 'T')
        W_list = []
        b_list = []
        Z_list = []

        print(f"Layers :> {self.layers}")

        for i in range(len(self.weights)):
            W_list.append( tf.Variable(self.weights[i].astype(np.float32)) )
            b_list.append( tf.Variable(self.bias[i].astype(np.float32)) )
            if i == 0:
                Z_list.append( tf.nn.relu( tf.matmul( X,W_list[i] ) + b_list[i]) )
            else:
                if i != len(self.weights) - 1:
                    Z_list.append( tf.nn.relu( tf.matmul( Z_list[i-1],W_list[i] ) + b_list[i]) )
                else:    
                    Z_list.append( tf.matmul( Z_list[i-1],W_list[i] ) + b_list[i] )

        cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits( logits = Z_list[-1], labels = T) )

        train_model_op = tf.train.RMSPropOptimizer(self.l_rate, self.decay, self.momentum).minimize(cost)
        predict = tf.argmax(Z_list[-1], 1)
        
        init = tf.initialize_all_variables()

        with tf.Session() as s:
            s.run(init)
            for i in range(self.n_iter):
                for j in range(n_batch):
                    Xbatch = self.X_train[ j*self.b_size : (j+1)*self.b_size, ]
                    Ybatch = Y_train_ind[ j*self.b_size : (j+1)*self.b_size, ]
                    s.run(train_model_op, feed_dict = {X: Xbatch, T: Ybatch})
                    if not (j%10):
                        current_cost = s.run(cost, feed_dict = {X: self.X_test, T: Y_test_ind})
                        predictions = s.run(predict, feed_dict = {X: self.X_test})
                        error = self.error_rate(predictions, self.Y_test)
                    print(f"Iteration of {i} in {j}th batch : [{i},{j}] Cost : {current_cost:.3f} Error: {error:.3f}")
                    self.test_error.append(error)
            train_prediction = s.run(predict, feed_dict = { X: self.X_train })
            self.final_training_error = self.error_rate(train_prediction, self.Y_train)

        print(f"Final Training Error is {self.final_training_error}")
        print(f"Final Test Error is {self.test_error[-1]}")

        plt.plot(self.test_error, label = "Error rate")
        plt.title(f"Error Rate of Model on Test Data in {self.n_iter} Iterations")

        # Reseting for another model
        history = {"model_id" : self.modelNo, "test_error": self.test_error, "layers": self.layers}
        self.pastModels.append(history)

        self.layers = []
        self.weights = []
        self.bias = []
        self.test_error = []
        


    