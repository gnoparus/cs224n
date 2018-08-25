#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(X, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    the backward propagation for the gradients for all parameters.

    Notice the gradients computed here are different from the gradients in
    the assignment sheet: they are w.r.t. weights, not inputs.

    Arguments:
    X -- M x Dx matrix, where each row is a training example x.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

#     print('params.shape = ', params.shape)
    
    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Note: compute cost based on `sum` not `mean`.
    ### YOUR CODE HERE: forward propagation
        
    h = sigmoid(X.dot(W1) + b1)    
    yhat = softmax(h.dot(W2) + b2)    
          
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    gradW1 = np.zeros_like(W1)
    gradb1 = np.zeros_like(b1)
    gradW2 = np.zeros_like(W2)
    gradb2 = np.zeros_like(b2)

#     print('W1.shape = ', W1.shape)
#     print('b1.shape = ', b1.shape)
#     print('W2.shape = ', W2.shape)
#     print('b2.shape = ', b2.shape)
   
    M = X.shape[0]
#     print('h.shape = ', h.shape)
    
    # cross entropy loss
    cost = -np.sum(labels * np.log(yhat)) / M
    
#     print('cost = ', cost)
    
#     print('yhat.shape = ', yhat.shape)
#     print('yhat[0] = ', yhat[0])
#     print('labels.shape = ', labels.shape)
#     print('labels[0] = ', labels[0])
#     print('labels = ', labels)
#     print('labels sum = ', np.sum(labels, axis=1, keepdims=True))
    dyhat = (yhat - labels) / M
#     print('dyhat.shape = ', dyhat.shape)
    
    gradb2 = np.sum(dyhat, axis=0, keepdims=True) 
    gradW2 = h.T.dot(dyhat) 

#     print('dyhat.dot(W2.T) = ', dyhat.dot(W2.T))
    
    dh = sigmoid_grad(h) * dyhat.dot(W2.T)
#     print('dh = ', dh)
    gradb1 = np.sum(dh, axis=0, keepdims=True) 
    gradW1 = X.T.dot(dh)
    
#     print('dh.shape = ', dh.shape)
#     print('gradb2.shape = ', gradb2.shape)
#     print('gradW2.shape = ', gradW2.shape)
#     print('gradb1.shape = ', gradb1.shape)
#     print('gradW1.shape = ', gradW1.shape)

    ### END YOUR CODE

    ### Stack gradients (do not modify)
#     grad = np.concatenate((gradb2.flatten(), gradW2.flatten(), gradb1.flatten(), gradW1.flatten()))
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print ("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in np.arange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print ("Running your sanity checks...")
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
