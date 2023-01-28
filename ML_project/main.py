import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def filter_56(x, y):
    keep = (y == 5) | (y == 6)
    x, y = x[keep], y[keep]
    return x,y

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def ComputeCostGrad(X, y, theta, _lambda):
    
    h = sigmoid(np.dot(X,theta))
    
    reg = (_lambda / (2.0 ) ) * np.sum(theta**2)#reguralization
    cur_j = (np.dot(y.T,np.log(h))+np.dot((1-y).T,np.log(1-h))) - reg
    
    reg = _lambda * theta# /(m+0.0)
    grad = np.dot(X.T,(y-h)) - reg
        
    return cur_j, grad

def predict(theta, X):
    #PREDICT Predict whether the label is 0 or 1 using learned logistic 
    #regression parameters theta
    #   p = PREDICT(theta, X) computes the predictions for X using a 
    #   threshold at 0.5 (i.e., if sigmoid( X * theta ) >= 0.5, predict 1)

    m = X.shape[0]
    
    #You need to return the following variables correctly
    p = np.zeros( (m,1) );
    p = sigmoid( np.dot(X,theta) )
    prob = p
    p = p > 0.5 - 1e-6

    return p, prob

def ComputeLogisticRegression(X, y, X_val, y_val, _lambda=0.0, tot_iter=1800, alpha=0.1):
 
    theta = np.zeros(X.shape[1]).reshape((-1,1))
    m, n = X.shape
    
    J_train = []
    J_test = []
    
    for i in range( tot_iter ):    

        train_error, train_grad = ComputeCostGrad(X, y, theta, _lambda)
        test_error, _ = ComputeCostGrad(X_val, y_val, theta, _lambda)
       
        #update parameters by adding gradient values (ascent)
        theta = theta + (alpha * train_grad/m)

        # print test error to validate the algorithm converges
        # print( test_error[0])
        
        #store current cost
        J_train.append(train_error[0])
        J_test.append(test_error[0])
        
    return J_train, J_test, theta

def main():
   
    # load mnist dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # x_train : (60000,28,28)
    # filter the examples with class label 5 or 6
    X_train, Y_train = filter_56(x_train, y_train)
    x_test, y_test = filter_56(x_test, y_test)
    
    # Map to classes {5, 6} -> {0, 1}
    Y_train -= 5
    y_test -= 5

    # split training data to 80% training data 20% validation data
    x_val, x_train, y_val, y_train = train_test_split(X_train, Y_train, test_size=0.8, random_state=42)

    print("Train set size: ", len(x_train))
    print("Test set size: ", len(x_test))
    print("Valid set size: ", len(x_val)) 

    # convert every image to 784
    x_train = x_train.reshape(x_train.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)
    x_val = x_val.reshape(x_val.shape[0], 784)
    
    # convert all data in vector to [0,1]
    x_train = x_train.astype(float) / 255
    x_test = x_test.astype(float) / 255
    x_val = x_val.astype(float) / 255

    # Add ones as first feature
    x_train = np.hstack([np.ones((x_train.shape[0],1)), x_train])
    x_test = np.hstack([np.ones((x_test.shape[0],1)), x_test])
    x_val = np.hstack([np.ones((x_val.shape[0],1)), x_val])

    # Transform y variables to column vectors (1d array) 
    y_train = y_train.reshape(-1,1).astype(int)  
    y_test = y_test.reshape(-1,1).astype(int) 
    y_val = y_val.reshape(-1,1).astype(int) 

    J_train, J_test, theta = ComputeLogisticRegression(x_train, y_train, x_val, y_val, _lambda=0.0 )

    plt.figure(figsize=(10,4))
    plt.subplot( 1, 2, 1 )
    plt.plot(np.arange( len(J_train) ), J_train, label='λ=0')
    plt.xlabel('Number of iterations')
    plt.ylabel('Train Error')
    plt.legend()
    plt.subplot( 1,2,2)
    plt.plot( np.arange( len(J_test) ), J_test, label='λ=0')
    plt.xlabel('Number of iterations' )
    plt.ylabel('Test error')
    plt.legend()
    plt.show()
    p_train, prob_train = predict( theta, x_train )
    p_test, prob_test = predict( theta, x_test )
    print( 'Accuracy of training set', np.mean( p_train.astype('int') == y_train ) )
    print( 'Accuracy of testing set', np.mean( p_test.astype('int') == y_test ) )

    print("After regularization")
    lambdas = np.linspace(0.001, 10, 10)
    train_stats, test_stats = [], []
    train_scores =[]
    test_scores = []

    for value in lambdas:
        J_train, J_test, theta = ComputeLogisticRegression( x_train, y_train, x_val, y_val, _lambda=value )
        train_stats.append( {'history' : J_train, 'lambda': value} )
        test_stats.append( {'history' : J_test, 'lambda': value} )

        p_train, prob_train = predict( theta, x_train )
        p_test, prob_test = predict( theta, x_test )
        train_scores.append(np.mean( p_train.astype('int') == y_train ))
        test_scores.append(np.mean( p_test.astype('int') == y_test ))

    # Plot the error of train, test per lambda value
    for i in range( len(train_stats) ):
        plt.plot( np.arange( len(train_stats[i]['history']) ), train_stats[i]['history'], label='λ=' + str( train_stats[i]['lambda'] ) )
    plt.xlabel( 'Number of iterations' )
    plt.ylabel( 'Train Error' )
    plt.legend()
    plt.show()

    for i in range( len(test_stats) ):
        plt.plot( np.arange( len(test_stats[i]['history']) ), test_stats[i]['history'], label='λ='+ str( test_stats[i]['lambda'] ) )
    plt.xlabel( 'Number of iterations' )
    plt.ylabel( 'Test error' )
    plt.legend()
    plt.show()


    plt.plot(lambdas,train_scores)
    plt.xlabel('Lambda')
    plt.ylabel('Train Accuracy')
    plt.show()

    plt.plot(lambdas,test_scores)
    plt.xlabel('Lambda')
    plt.ylabel('Test Accuracy')
    plt.show()


main()



