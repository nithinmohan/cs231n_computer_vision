import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  num_train = X.shape[0]
  dW = np.zeros_like(W)
  margins = X.dot(W)
  exponentiated = np.exp(margins)
  correct = exponentiated[np.arange(num_train),y]
  sum = np.sum(exponentiated,axis=1)
  #print(correct.shape)
  #print(sum.shape)
  ratio = np.divide(correct,sum)
  #print(ratio)
  log = np.log(ratio)*-1
  
  
  loss = np.mean(log)
  loss+=np.sum(W*W)*reg
    
  xmask = exponentiated/sum[:,np.newaxis]
  #print(xmask)
  xmask[np.arange(num_train), y] -= 1
  dW = X.T.dot(xmask)
  dW= np.divide(dW,num_train)
  #dW+=(2*reg*W)

  return(loss,dW)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.

  # Initialize the loss and gradient to zero.
  loss = 0.0
  num_train = X.shape[0]
  dW = np.zeros_like(W)
  margins = X.dot(W)
  #margins -= np.max(margins, axis=1, keepdims=True) # max of every sample
    
  exponentiated = np.exp(margins)
  #correct = exponentiated[np.arange(num_train),y]
  sum = np.sum(exponentiated,axis=1,keepdims=True)
  #print(correct.shape)
  #print(sum.shape)
  ratio = np.divide(exponentiated,sum)
  #print(ratio)
  log = np.log(ratio[np.arange(num_train), y])*-1
  #loss = np.sum(log)
  
  loss = np.mean(log)
  loss+=np.sum(W*W)*reg
    
  xmask = ratio
  #print(xmask)
  xmask[np.arange(num_train), y] -= 1
  dW = X.T.dot(xmask)
  dW= np.divide(dW,num_train)
  dW+=(2*reg*W)

  return(loss,dW)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

