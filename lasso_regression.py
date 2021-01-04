import numpy as np


'''The cost function for lasso regression which is which is: (1/2*m) * ∑((h(x) - y) ^ 2) + L1 * w
   where h(x) = predicted output and y = actual output
   and m is the number of training examples'''


def cost_func_lasso(X_train,y_train,w, L1):
#Using the MSE cost function which is the squared difference between the actual and predicted values with the L1 penalty over the weights
    return (np.sum((X_train.dot(w) - y_train) ** 2) + L1 *np.sum(w))/(2 * len(y_train))        

    
'''Performing gradient descent to minimize the sum of squared errrors by calculating
   the gradient of the cost function i.e. partial derivative which is x(x*w-y) + lambda * sign(w)
   In this case, we use L1 regularization which uses the L1 norm as a penalty term to the weights
'''

def gradient_descent_lasso(X_train, y_train, w, lr, epoch, L1):
    costs = []
    tolerance = 0.0001               #Setting up a tolerance value which checks for significant decrease in weights for early stopping
    i = 0
    cost = cost_func_lasso(X_train,y_train,w,L1)
    costs.append(cost)
    flag = 1
    stop = 0
    
    for epoch in range(1,epoch):
    
        for i in range(0,len(X_train),50):
            X_train1 = X_train[i:i+50]                    #Implementing Mini-Batch Gradient Descent with mini-batches of 50 to improve speed of convergence
            y_train1 = y_train[i:i+50]
            loss = (X_train1.dot(w)) - y_train1           #Calculating the difference between the predicted and actual value
            
            gradient = (X_train1.T.dot(loss) + L1*np.sign(w)) / len(y_train1)    #x(x*w-y) + lambda * sign(w) which is the partial derivative i.e. gradient of the cost function
            new_w = w - (lr * gradient)

            if np.sum(abs(new_w - w)) < tolerance:            #Checking early stopping condition
                flag = 0
                break

            w = new_w
            
        cost = cost_func_lasso(X_train,y_train,w, L1)
        costs.append(cost)
        
        if costs[epoch] > costs[epoch-1] : stop += 1          #Check if the Cost function increases
        else: stop = 0
            
        if flag ==0 or stop > 5: break
        
    return w, costs