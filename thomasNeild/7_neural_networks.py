"""7-1. plotting the relu function"""
from sympy import *

# plot relu
x = symbols('x')
relu = Max(0, x)
# plot(relu)

"""7-2. logistic (sigmoid) activation function in sympy"""
# plot logistic (sigmoid)
logistic = 1 / (1 + exp(-x))
# plot(logistic)

"""7-3. a simple forward propagation network with random weight and bias values"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

rgb_data = pd.read_csv("data/light_dark_font_training_set.csv")

# grab independent variable columns (all rows, all columns but last column)
# .values method returns data points only (no index and header)
# scale down by 255
inputs = (rgb_data.iloc[:, 0:3].values / 255.0)

# grab dependent "LIGHT_OR_DARK_FONT_IND" variable column (all rows, last column)
output = rgb_data.iloc[:, -1].values

# split train and test data sets
X_train, X_test, Y_train, Y_test = train_test_split(inputs, output, test_size=1/3)
n = X_train.shape[0] # number of training records

# build a neural network with weights and biases
# with random initialization
hidden_weights = np.random.rand(3, 3) # 3 sets of 3 random numbers
hidden_biases = np.random.rand(3, 1) # 3 sets of 1 random number

output_weights = np.random.rand(1, 3) # 1 set of 3 random numbers
output_bias = np.random.rand(1, 1) # 1 set of 1 random number

# activation functions
relu = lambda x: np.maximum(x, 0)
logistic = lambda x: 1 / (1 + np.exp(-x))

# runs inputs through the neural network to get predicted outputs
def forward_prop(X):
  input_layer_output = hidden_weights @ X + hidden_biases
  relu_activation_output = relu(input_layer_output)
  hidden_layer_output = output_weights @ relu_activation_output + output_bias
  logistic_activation_output = logistic(hidden_layer_output)
  return (input_layer_output,         # Z1
          relu_activation_output,     # A1
          hidden_layer_output,        # Z2
          logistic_activation_output) # A2

# calculate accuracy
test_predictions = forward_prop(X_test.transpose())[3] # grab only logistic activation output
test_comparisons = np.equal((test_predictions >= .5).flatten().astype(int), Y_test) # >= .5 suggests light font and < .5 suggests dark font
accuracy = sum(test_comparisons.astype(int) / X_test.shape[0])
# print("ACCURACY: ", accuracy)

# OUTPUT
# ACCURACY:  0.623608017817371

"""7-7. calculating the derivative of the cost function with respect to the logistic activation output"""
A2, Y = symbols('A2 Y') # A2 here is the logistic activation output
C = (A2 - Y)**2         # C here is the cost function
dC_dA2 = diff(C, A2)
# print(dC_dA2) # 2*A2 - 2*Y

"""7-8. finding the derivative of A2 with respect to Z2"""
Z2 = symbols('Z2') #Z2 here is the hidden layer output

_logistic = lambda x: 1 / (1 + exp(-x))

A2 = _logistic(Z2)
dA2_dZ2 = diff(A2, Z2)
# print(dA2_dZ2) # exp(-Z2)/(1 + exp(-Z2))**2

"""7-9. derivative of Z2 with respect to W2"""
A1, W2, B2 = symbols('A1, W2, B2')  # A1 here is the relu activation output
                                    # W2 here are the output weights
                                    # B2 here is the output bias

Z2 = A1*W2 + B2
dZ2_dW2 = diff(Z2, W2)
# print(dZ2_dW2) # A1

"""7-10. finish calculating all partial derivatives we will need for our neural network"""
W1, B1, Z1, X = symbols('W1 B1 Z1 X')

# derivative of Z2 with respect to B2
dZ2_dB2 = diff(Z2, B2)
# print(dZ2_dB2) # 1

# derivative of A1 with respect to Z1
_relu = lambda x: Max(x, 0)
A1 = _relu(Z1)

d_relu = lambda x: x > 0 # slope is 1 if positive, 0 if negative
dA1_dZ1 = d_relu(Z1)
# print(dA1_dZ1) # Z1 > 0

# derivative of Z1 with respect to W1
Z1 = X*W1 + B1
dZ1_dW1 = diff(Z1, W1)
# print(dZ1_dW1) # X

# derivative of Z1 with respect to B1
dZ1_dB1 = diff(Z1, B1)
# print(dZ1_dB1) # 1

"""7-11. implementing a neural network using stochastic gradient descent"""
# learning rate controls how slowly we approach a solution
# make it too small, it will take too long to run
# make it too big, it will likely overshoot and miss the solution
learning_rate = 0.05

# derivatives of activation functions
d_relu = lambda x: x > 0
d_logistic = lambda x: np.exp(-x) / (1 + np.exp(-x)) ** 2

# returns slopes for weights and biases
# using chain rule
def backward_prop(Z1, A1, Z2, A2, X, Y):
  dC_dA2 = 2 * A2 - 2 * Y
  dA2_dZ2 = d_logistic(Z2)
  dZ2_dA1 = output_weights
  dZ2_dW2 = A1
  dZ2_dB2 = 1
  dA1_dZ1 = d_relu(Z1)
  dZ1_dW1 = X
  dZ1_dB1 = 1

  dC_dW2 = dC_dA2 @ dA2_dZ2 @ dZ2_dW2.T

  dC_dB2 = dC_dA2 @ dA2_dZ2 * dZ2_dB2

  dC_dA1 = dC_dA2 @ dA2_dZ2 @ dZ2_dA1

  dC_dW1 = dC_dA1 @ dA1_dZ1 @ dZ1_dW1.T

  dC_dB1 = dC_dA1 @ dA1_dZ1 * dZ1_dB1

  return dC_dW1, dC_dB1, dC_dW2, dC_dB2

# number of iterations to perform gradient descent
iterations = 100_000 # often set to 1_000 iterations (Starmer, p. 95)

# execute gradient descent
"""
for i in range(iterations):
  # randomly select one of the training data
  idx = np.random.choice(n, 1, replace=False)
  X_sample = X_train[idx].transpose()
  Y_sample = Y_train[idx]

  # run randomly selected training data through neural network
  Z1, A1, Z2, A2 = forward_prop(X_sample)

  # distribute error through backpropagation
  # and return slopes for weights and biases
  dW1, dB1, dW2, dB2 = backward_prop(Z1, A1, Z2, A2, X_sample, Y_sample)

  # update weights and biases
  hidden_weights -= learning_rate * dW1
  hidden_biases -= learning_rate * dB1
  output_weights -= learning_rate * dW2
  output_bias -= learning_rate * dB2
"""

# Calculate accuracy
test_predictions = forward_prop(X_test.transpose())[3] # grab only logistic activation output
test_comparisons = np.equal((test_predictions >= .5).flatten().astype(int), Y_test) # >= .5 suggests light font and < .5 suggests dark font
accuracy = sum(test_comparisons.astype(int) / X_test.shape[0])
# print("ACCURACY: ", accuracy)

# OUTPUT
# ACCURACY:  0.9844097995545583

# save these values so epochs can be skipped when the script is re-run
# print(hidden_weights, hidden_biases, output_weights, output_bias)
from numpy import array

hidden_weights = array([[4.02725949, 7.90954798, 1.69030231],
                        [4.12124653, 8.11815365, 1.46076542],
                        [4.40104025, 8.32754953, 1.21201887]])
hidden_biases = array([ [-6.38508591],
                        [-6.09658981],
                        [-6.36471365] ])
output_weights = array([[3.29431125, 3.41760165, 4.02731699]])
output_bias = array([[-4.67371176]])

"""7-12. adding an interactive shell to our neural network"""
# interact and test with new colors
def predict_probability(r, g, b):
  X = np.array([[r, g, b]]).transpose() / 255
  Z1, A1, Z2, A2 = forward_prop(X)
  return A2

def predict_font_shade(r, g, b):
  output_values = predict_probability(r, g, b)
  if output_values > .5:
      return "DARK"
  else:
      return "LIGHT"

# while True:
#   col_input = input("Predict light or dark font. Input values R,G,B: ")
#   (r, g, b) = col_input.split(",")
#   print(predict_font_shade(int(r), int(g), int(b)))

"""7-13. using scikit-learn neural network classifier"""
from sklearn.neural_network import MLPClassifier

neural_network = MLPClassifier( solver='sgd',
                                hidden_layer_sizes=(3, ),
                                activation='relu',
                                max_iter=iterations,
                                learning_rate_init=learning_rate)

neural_network.fit(X_train, Y_train)

# print weights and biases
# print(neural_network.coefs_)
# print(neural_network.intercepts_)

# OUTPUT
"""
[array([[-0.14714534,  0.7305387 ,  3.22428062],
       [ 0.13318895,  1.19124688,  6.66936814],
       [ 0.52467993, -0.30319235,  1.29417808]]), array([[-0.39836948],
       [ 0.70095979],
       [ 8.92776468]])]
[array([-0.77645723, -0.45519071, -4.95952648]), array([-5.73372202])]
"""

# print(f"Training set score: {neural_network.score(X_train, Y_train)}")
# print(f"Test set score: {neural_network.score(X_test, Y_test)}")

# OUTPUT
# Training set score: 0.9977678571428571
# Test set score: 0.9955456570155902

"""Exercises"""
from sklearn.metrics import confusion_matrix

employee_data = pd.read_csv("data/employee_retention_analysis.csv", delimiter=",")

# grab independent variable columns (all rows, all columns but last column)
# .values method returns data points only (no index and header)
X = employee_data.values[:, :-1]

# grab dependent "did_quit" variable column (all rows, last column)
Y = employee_data.values[:, -1]

# separate training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.33, random_state=10)

neural_network = MLPClassifier( solver='sgd',
                                hidden_layer_sizes=(3, ),
                                activation='relu',
                                max_iter=iterations,
                                learning_rate_init=learning_rate)

neural_network.fit(X_train, Y_train)

# print(f"Training set score: {neural_network.score(X_train, Y_train)}")
# print(f"Test set score: {neural_network.score(X_test, Y_test)}")

# OUTPUT
# Training set score: 0.5555555555555556
# Test set score: 0.5

matrix = confusion_matrix(y_true=Y_test, y_pred=neural_network.predict(X_test))
# print("Confusion matrix:\n", matrix)

# OUTPUT
"""
Confusion matrix:
 [[9 0]
 [9 0]]
"""
