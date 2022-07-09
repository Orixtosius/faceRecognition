import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from get_data import dataManipulation

class Model():
    def __init__(self, l_rate, n_iter, reg, data) -> None:
        self.l_rate = l_rate
        self.n_iter = n_iter
        self.reg = reg
        
        data = dataManipulation()
        data.get_processed_data()
        self.X_train = data.X_train
        self.Y_train = data.Y_train
        self.X_test = data.X_test
        self.Y_test = data.Y_test

    def indicatorMatrix(y):
        N = len(y)
        y = y.astype(np.int32)
        K = y.max() + 1
        ind = np.zeros((N,K))
        for n in range(N):
            k = y[n]
            ind[n,k] = 1
        return ind

    def forwardPropagation(X,W,b):
        Z = X.dot(W) + b
        expZ = np.exp(Z)
        return expZ / expZ.sum(axis = 1, keepdims = True)

    def cost(p_y, t):
        tot = t * np.log(p_y)
        return -tot.sum()

    def predict(p_y):
        return np.argmax(p_y, axis = 1)

    def error_rate(p_y, t):
        prediction = predict(p_y)
        return np.mean(prediction != t)


    def gradW(t, y, X):
        return X.T.dot(t - y)

    def gradb(t, y):
        return (t - y).sum(axis = 0)


    def linearModel(path, index, split_ratio = 0.7, shuffle = 1, learning_rate = 0.0001, regularization = 0.0, n_iter = 100):

        train_x, train_y, test_x, test_y = get_data(path, index, split_ratio, shuffle)

        M, NX = train_x.shape
        test_indicatorMatrix = indicatorMatrix(test_y)
        train_indicatorMatrix = indicatorMatrix(train_y)
        K = train_indicatorMatrix.shape[1]
        print(f"Target number is {K}")
        print(f"Train ind matrix is : {train_indicatorMatrix.shape}")
        print(f"Test ind matrix is : {train_indicatorMatrix.shape}")

        print("Initialization of weight matrix and bias vector has started...\n")
        W = np.random.randn(NX,K) / np.sqrt(NX)
        print(f"Shape of W is {W.shape}")
        b = np.zeros(K)
        print(f"Shape of b is {b.shape}")
        print("\nInitialization of weight matrix and bias vector has COMPLETED\n")

        train_losses, test_losses, train_error, test_error = [],[],[],[]

        print("Training has started.\n")

        for i in range(n_iter):

            p_y = forwardPropagation(train_x, W, b)
            trainLoss = cost(p_y, train_indicatorMatrix)
            train_losses.append(trainLoss)
            trainError = error_rate(p_y, train_y)
            train_error.append(trainError)

            p_y_test = forwardPropagation(test_x, W, b)
            testLoss = cost(p_y_test, test_indicatorMatrix)
            test_losses.append(testLoss)
            testError = error_rate(p_y_test, test_y)
            test_error.append(testError)

            W += learning_rate*(gradW( train_indicatorMatrix, p_y, train_x) - regularization*W)
            b += learning_rate*(gradb( train_indicatorMatrix, p_y ))

            if (i+1)%10 == 0:
                print(f"Iteration : {i+1} Train Loss : {trainLoss:.3f} Train Error : {trainError:.3f}")
                print(f"                  Test Loss  : {testLoss:.3f}  Test Error  : {testError:.3f}")

        p_y = forwardPropagation(test_x, W, b)

        print("Training has finished.\n")
        print(f"Final error rate is {error_rate(p_y, test_y)}")

        results = {
            "test_error_rate": error_rate(p_y, test_y),
            "test_losses": test_losses,
            "costs": train_losses,
            "errors": train_error,
            "w" : W,
            "b" : b,
            "n_iter" : n_iter,
        }

        return results

def plot(results):
    for c,result in enumerate(results):
        plt.plot(result["costs"], label = f"Cost of Model {c}")
        plt.plot(result["test_losses"])
    plt.title("Loss per iteration")
    plt.legend()
    plt.show()

    