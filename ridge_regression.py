import numpy as np


'''The cost function for lasso regression which is which is: (1/2*m) * ∑((h(x) - y) ^ 2) + L2 * (w^2)
   where h(x) = predicted output and y = actual output
   and m is the number of training examples'''


def cost_func_ridge(X_train,y_train,w,L2):

#Using the MSE cost function which is the squared difference between the actual and predicted values along with a L2 regularization penalty to the weights
    return (np.sum((X_train.dot(w) - y_train) ** 2) + L2 * np.dot(w.T, w))/(2 * len(y_train))     
    
    
'''Performing gradient descent to minimize the sum of squared errrors by calculating
   the gradient of the cost function i.e. partial derivative which is x(x*w-y) + lambda * w
   In this case, we use L2 regularization which uses the L2s norm as a penalty term to the weights
'''

def gradient_descent_ridge(X_train, y_train, w, lr, epoch, L2):
    costs = []
    tolerance = 0.0001              #Setting up a tolerance value which checks for significant decrease in weights for early stopping
    i = 0
    flag = 1
    stop = 0
    cost = cost_func_ridge(X_train,y_train,w,L2)
    costs.append(cost)
    
    for epoch in range(1,epoch+1):
        for i in range(0,len(X_train),50):                #Implementing Mini-Batch Gradient Descent with mini-batches of 50 to improve speed of convergence
            X_train1 = X_train[i:i+50]
            y_train1 = y_train[i:i+50]
            loss = (X_train1.dot(w)) - y_train1           #Calculating the difference between the predicted and actual value

            gradient = (X_train1.T.dot(loss) + L2*w) / len(y_train1)     #x(x*w-y) + lambda * w which is the partial derivative i.e. gradient of the cost function

            new_w = w - (lr * gradient)

            if np.sum(abs(new_w - w)) < tolerance:            #Checking early stopping condition
                flag = 0
                break


            w = new_w
            
        cost = cost_func_ridge(X_train,y_train,w, L2)    
        costs.append(cost)
        
        if costs[epoch] > costs[epoch-1] : stop += 1           #Check if the Cost function increases
        else: stop = 0
            
        if flag ==0 or stop > 10:                              #Early Stopping
            break
            
                

        
    return w, costs