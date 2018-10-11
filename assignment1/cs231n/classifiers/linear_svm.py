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
  num_classes = W.shape[1] # C个类
  num_train = X.shape[0]   # N个训练样本数
  loss = 0.0
  for i in xrange(num_train): # 一次计算每个样本，每个 样本计算C次
    scores = X[i].dot(W)   # 1*D dot D*C
    correct_class_score = scores[y[i]] # y[i] 样本标签的类，correct_class_score 计算之后正确的值
    for j in xrange(num_classes):
      if j == y[i]:  # 如果是正确分的类，跳过
        # 计算 j = i 时的梯度
        # dW[:,y[i]] +=-X [i]
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:  # max(0, sj-syi+1)
        loss += margin
        #计算 j != y[i] 时的梯度
        dW[:,j] += X[i]
        # dW[:,y[i]] +=-X [i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  #
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * W
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
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # compute the loss and the gradient
  num_classes = W.shape[1] # C个类
  num_train = X.shape[0]   # N个训练样本数

  scores = X.dot(W)# N*C
  # print("scores")
  # print(scores)
  correct_class_score = scores[np.arange(num_train), y] #取正确分类分数
  # print("correct_class_score")
  # print(correct_class_score)

  #重复10次,得到500*10的矩阵,这样可以和scores相加相减
  correct_class_score = np.reshape(np.repeat(correct_class_score,num_classes),(num_train,num_classes))
  # print(correct_class_score)
  margin = scores-correct_class_score+1.0 # 500*10
  # print("margin1")
  # print(margin)
  margin[np.arange(num_train),y]=0 # j = yi 设为0 

  # print("margin2")
  # print(margin)

  loss = (np.sum(margin[margin > 0]))/num_train
  # print(margin[margin > 0])
  # print("loss")
  # print(loss)
  loss += 0.5*reg*np.sum(W*W)


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
  #gradient
  margin[margin>0]=1
  margin[margin<=0]=0

  # print("margin3")
  # print(margin)
  row_sum = np.sum(margin, axis=1)                  # 1 by N
  # print("row_sum")
  # print(row_sum)
  margin[np.arange(num_train), y] = -row_sum
  # print("margin4")
  # print(margin)
  dW += np.dot(X.T, margin)     # D by C

  #
  dW/=num_train
  dW += reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
