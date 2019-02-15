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
  dW = np.zeros_like(W)

  num_dim = W.shape[0]
  num_classes = W.shape[1]
  num_train = X.shape[0]

  for x in xrange(num_train):
      scores = np.dot(X[x,:], W)
      normal = scores - np.max(scores)
      softmax = np.exp(normal)/np.sum(np.exp(normal))
      loss += -np.log(softmax[y[x]])

      for ha in xrange(num_dim):
          for i in xrange(num_classes):
              if i == y[x]:
                  dW[ha,i] += X.T[ha,x] * (-1 + softmax[i]) # 여기가 왜 이거냐...?!
              else :
                  dW[ha,i] += X.T[ha,x] * softmax[i]


  loss /= num_train
  loss += 0.5 * reg * np.sum(W**2)

  dW /= num_train
  dW += reg * W
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
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
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]

  scores = np.dot(X, W)
  scores -= np.max(scores,axis=1)[...,np.newaxis]
  softmax = np.exp(scores)/np.sum(np.exp(scores),axis=1)[...,np.newaxis]

  dS = softmax
  dS[np.arange(num_train),y] -= 1
  dW = X.T.dot(dS)

  correct_class_score = np.choose(y, scores.T) #scores[np.arange(num_train),y] 랑 뭐가 다르지..?
  #동일한 것 같은데 결과가 다르다 이유가 뭐야>??뭐야????? 모르겠...

  term = np.log(np.sum(np.exp(scores), axis=1))
  loss = -correct_class_score + term
  loss = np.sum(loss)

  loss /= num_train
  loss += 0.5* reg * np.sum(W*W)

  dW /= num_train
  dW += reg* W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
