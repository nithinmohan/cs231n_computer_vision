import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    wrong_c = np.zeros(W.shape[1])
    count = 0
    for j in xrange(num_classes):   
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        count+=1
        wrong_c[j]=1
        loss += margin
        #dW += X[i]*wrong_c[:,np.newaxis]*count
    tdW = X[i][:,np.newaxis]*wrong_c
    tdW[:,y[i]] = X[i]*count*-1
    dW+=tdW
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW = np.divide(dW,num_train)
  dW+=reg*W
  # Add regularization to the loss.
  loss += 2*reg*np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.ones(W.shape) # initialize the gradient as zero
  #print(W.shape)
  num_train = X.shape[0]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  correct = scores[np.arange(scores.shape[0]),y]
  margin = scores-correct[:,np.newaxis]+1
  margin[np.arange(margin.shape[0]),y] = 0
  margin[margin<0]=0
  loss = np.sum(margin)
  loss/=num_train
  loss += reg * np.sum(W * W)
  #dW=np.ones(W.shape)
  #copied
  # Fully vectorized version. Roughly 10x faster.
  X_mask = np.zeros(margin.shape)
  X_mask[margin > 0] = 1
  # for each sample, find the total number of classes where margin > 0
  incorrect_counts = np.sum(X_mask, axis=1)
  X_mask[np.arange(num_train), y] = -incorrect_counts
  dW = X.T.dot(X_mask)

  #dW=np.tile(X,(10,1,1))
  #margin=margin.T
  #dW[(margin<=0)]=0
  #dW[y,np.arange(margin.shape[1])]=0
  #dW[y,np.arange(margin.shape[1])]=np.sum(dW,axis=0)*-1
  #dW=np.sum(dW,axis=1).T
  dW= np.divide(dW,num_train)
  
  dW+=(2*reg*W)
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
