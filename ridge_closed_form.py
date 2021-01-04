import numpy as np

'''Implementation of the closed form solution of Ridge regression
    inv(X*X + λ*I) * X*Y'''
    

def solver(X_train,y_train,L2): 
    I = np.eye((X_train.shape[1]))      #Defining 'I' i.e. the identity matrix
    return (np.linalg.inv(X_train.T @ X_train + L2 * I) @ X_train.T @ y_train)      #Closed form solution for Ridge Regression
    
    #@ provides matrix multiplcation