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
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        #이거 잘 생각해봐 j랑 y[i]가 뭔지,,
        dW[:,y[i]] -= X[i,:]
        dW[:,j] += X[i,:]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg *W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW

#vectorized로 loss를 구하는 것이 조금 더 복잡하지만 즉각적으로 loss 결과를 사용할 수 있다는 장점
def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]

  scores = np.dot(X, W)

  correct_class_scores = scores[np.arange(num_train),y] # np.choose uses y to select elements from scores.T
    # Calculate our margins all at once.
  margin = np.maximum(0,scores - correct_class_scores[..., np.newaxis] + 1)

    # Only add margin to our loss if it's greater than 0, let's make
    # negative margins =0 so they dont change our loss.
  margin[margin < 0] = 0

    # Average our data loss over the size of batch and add reg. term to the loss.
  loss = np.sum(margin) / num_train
  #### 여기서 W*W를 의미하지않는다.... W.T에다가 W을 곱하는거다....ㅠ
  loss += reg * np.sum(W.T.dot(W))
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
  binary = margin
  binary[margin > 0] = 1
  row_sum = np.sum(binary, axis =1)
  binary[np.arange(num_train),y] = -row_sum.T
  dW = np.dot(X.T, binary)
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
