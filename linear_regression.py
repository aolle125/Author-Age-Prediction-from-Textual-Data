import numpy as np

'''The cost function for linear regression which is: (1/2*m) * ∑((h(x) - y) ^ 2)
   where h(x) = predicted output and y = actual output
   and m is the number of training examples'''
   
   
def cost_function(X_train,y_train,w):
#Using the MSE cost function which is the squared difference between the actual and predicted values 
    return np.sum((X_train.dot(w) - y_train) ** 2)/(2 * len(y_train))      
    
    
    
'''Performing gradient descent to minimize the sum of squared errrors by calculating
   the gradient of the cost function i.e. partial derivative which is x(x*w-y)
'''

def gradient_descent_linear(X_train, y_train, w, lr, epoch):
    costs = []
    tolerance = 0.001           #Setting up a tolerance value which checks for significant decrease in loss for early stopping
    cost = cost_function(X_train,y_train,new_w)
    costs.append(cost)
    i = 0
    for epoch in range(epoch):      
        loss = (X_train.dot(w)) - y_train               #Calculating the difference between the predicted and actual value
        
        gradient = (X_train.T.dot(loss))/ len(y_train)      #x(x*w-y) which is the partial derivative i.e. gradient of the cost function
        
        new_w = w - lr * gradient                       #Updating our weight values in the direction opposite to the gradient
        
        cost = cost_function(X_train,y_train,new_w)     #Calculating the mse loss
        
        if np.sum(abs(new_w - w)) < tolerance:          #Checking early stopping condition
            break
            
        costs.append(cost)
        
        
        if cost > costs[epoch-1]:                       #Checking if the cost function increases rather than decreases for some iterations
            i+=1
        if i > 10: break
        w = new_w
        
    return w, costs                                               


