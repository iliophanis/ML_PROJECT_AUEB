import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

#use by default ax=1, when the array is 2D
#use ax=0 when the array is 1D
def softmax( x, ax=1 ):
    m = np.max( x, axis=ax, keepdims=True )#max per row
    p = np.exp( x - m )
    return ( p / np.sum(p,axis=ax,keepdims=True) )

def view_dataset(x):
    # plot 5 random images from the training set
    n = 100
    sqrt_n = int( n**0.5 )
    samples = np.random.randint(x.shape[0], size=n)

    plt.figure( figsize=(11,11) )

    cnt = 0
    for i in samples:
        cnt += 1
        plt.subplot( sqrt_n, sqrt_n, cnt )
        plt.subplot( sqrt_n, sqrt_n, cnt ).axis('off')
        plt.imshow( x[i].reshape(28,28), cmap='gray'  )

    plt.show()

def cost_grad_softmax(W, X, t, lamda):#X: NxD, #W: KxD, #t: NxD
    
    E = 0
    N, D = X.shape
    K = t.shape[1]
    
    y = softmax( np.dot(X,W.T) )
    
    for n in range(N):
        for k in range(K):
            E += t[n][k] * np.log( y[n][k] )
    E -= lamda * np.sum( np.square( W ) ) / 2
    
    gradEw = np.dot( (t-y).T, X ) - lamda * W
    
    return E, gradEw

def ml_softmax_train(t, X, lamda, Winit, options):
    """inputs :
      t: N x 1 binary output data vector indicating the two classes
      X: N x (D+1) input data vector with ones already added in the first column
      lamda: the positive regularizarion parameter
      winit: D+1 dimensional vector of the initial values of the parameters
      options: options(1) is the maximum number of iterations
               options(2) is the tolerance
               options(3) is the learning rate eta
    outputs :
      w: the trained D+1 dimensional vector of the parameters"""

    W = Winit

    # Maximum number of iteration of gradient ascend
    _iter = options[0]

    # Tolerance
    tol = options[1]

    # Learning rate
    eta = options[2]

    Ewold = -np.inf
    costs = []
    m, n = X.shape
    for i in range( 1, _iter+1 ):
        Ew, gradEw = cost_grad_softmax(W, X, t, lamda)
        # save cost
        costs.append(Ew)
        # Show the current cost function on screen
        if i % 50 == 0:
            print('Iteration : %d, Cost function :%f' % (i, Ew))

        # Break if you achieve the desired accuracy in the cost function
        if np.abs(Ew - Ewold) < tol:
            break
                
        # Update parameters based on gradient ascend
        W = W + (eta * gradEw)

        Ewold = Ew

    return W, costs

def gradcheck_softmax(Winit, X, t, lamda):
    
    W = np.random.rand(*Winit.shape)
    epsilon = 1e-6
    
    _list = np.random.randint(X.shape[0], size=5)
    x_sample = np.array(X[_list, :])
    t_sample = np.array(t[_list, :])
    
    Ew, gradEw = cost_grad_softmax(W, x_sample, t_sample, lamda)
    
    print( "gradEw shape: ", gradEw.shape )
    
    numericalGrad = np.zeros(gradEw.shape)
    # Compute all numerical gradient estimates and store them in
    # the matrix numericalGrad
    for k in range(numericalGrad.shape[0]):
        for d in range(numericalGrad.shape[1]):
            
            #add epsilon to the w[k,d]
            w_tmp = np.copy(W)
            w_tmp[k, d] += epsilon
            e_plus, _ = cost_grad_softmax(w_tmp, x_sample, t_sample, lamda)

            #subtract epsilon to the w[k,d]
            w_tmp = np.copy(W)
            w_tmp[k, d] -= epsilon
            e_minus, _ = cost_grad_softmax( w_tmp, x_sample, t_sample, lamda)
            
            #approximate gradient ( E[ w[k,d] + theta ] - E[ w[k,d] - theta ] ) / 2*e
            numericalGrad[k, d] = (e_plus - e_minus) / (2 * epsilon)
    
    return ( gradEw, numericalGrad )

def ml_softmax_test(W, X_test):
    ytest = softmax( X_test.dot(W.T) )
    # Hard classification decisions
    ttest = np.argmax(ytest, 1)
    return ttest

def main():
    # load mnist dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Map to class {6} -> IF 6 THEN 1 ELSE 0 -> {0, 1}
    # y_train = [ 1 if a == 6 else 0 for i, a in enumerate(y_train) ]
    # y_test = [ 1 if a == 6 else 0 for i, a in enumerate(y_test) ]

    # split training data to 80% training data 20% validation data

    print("Train set size: ", len(x_train))
    print("Test set size: ", len(x_test))
    view_dataset(x_train)
    # convert every image to 784
    x_train = x_train.reshape(x_train.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)
    
    # convert all data in vector to [0,1]
    x_train = x_train.astype(float) / 255
    x_test = x_test.astype(float) / 255

    # Add ones as first feature
    x_train = np.hstack([np.ones((x_train.shape[0],1)), x_train])
    x_test = np.hstack([np.ones((x_test.shape[0],1)), x_test])

    # Transform y variables to column vectors as one-hot representation
    # eg. label:  0  in one-hot representation:  [1 0 0 0 0 0 0 0 0 0] 
    lr = np.arange(10)
    y_train= np.array([(lr==y_train[i]).astype('int') for i in range(len(y_train))])
    y_test =np.array([(lr==y_test[i]).astype('int') for i in range(len(y_test))])
    # y_train = np.asarray(y_train).reshape(-1,1).astype(int)  
    # y_test = np.asarray(y_test).reshape(-1,1).astype(int)

    # N of X
    N, D = x_train.shape

    K = 10

    # initialize w for the gradient ascent
    Winit = np.zeros((K, D))

    # regularization parameter
    lamda = 0.1

    # options for gradient descent
    options = [500, 1e-6, 0.5/N]

    gradcheck_softmax(Winit, x_train, y_train, lamda)

    # Train the model
    W, costs = ml_softmax_train(y_train, x_train, lamda, Winit, options)
    
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(format(options[2], 'f')))
    plt.show()

    pred = ml_softmax_test(W, x_test)
    print("Test Accuracy: ",np.mean( pred == np.argmax(y_test,1) ) * 100)
main()