"""5-1. using scikit-learn to do a linear regression"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# import points
df = pd.read_csv('data/single_independent_variable_linear_small.csv', delimiter=",")

# extract input variables (all rows, all columns but last column)
# .values method returns data points only (no index and header)
# use native accessor by tacking on [] 
X = df.values[:, :-1]

# extract output column (all rows, last column)
Y = df.values[:, -1]

# fit a line to the points
fit = LinearRegression().fit(X, Y)

m = fit.coef_.flatten()
b = fit.intercept_.flatten()
# print(f"m = {m}") # m = [1.93939394]
# print(f"b = {b}") # b = [4.73333333]

# show in chart
# plt.plot(X, Y, 'o')   # scatterplot
                        # 'o' - circle marker format
                        # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
# plt.plot(X, m*X + b)  # line
# plt.show()

"""5-2. calculating the residuals for a given line and data"""
# import points with .itertuples() so we can loop over the dataframe
points = pd.read_csv('data/single_independent_variable_linear_small.csv', delimiter=",").itertuples()

# test with a given line
m = 1.93939
b = 4.73333

# calculate the residuals
# for p in points:
#   y_actual = p.y
#   y_predict = m*p.x + b
#   residual = y_actual - y_predict
  # print(residual) 

"""5-4. calculating the sum of squares for a given line and data"""
sum_of_squares = 0.0

# calculate sum of squares
for p in points:
  y_actual = p.y
  y_predict = m*p.x + b
  residual_squared = (y_actual - y_predict)**2
  sum_of_squares += residual_squared

# print(f"sum of squares = {sum_of_squares}") 
# sum of squares = 28.096969704500005

"""5-5. calculating m and b for a simple linear regression"""
# load the data with list() and .itertuples()
points = list(pd.read_csv('data/single_independent_variable_linear_small.csv', delimiter=",").itertuples())

n = len(points)

m = (n * sum(p.x * p.y for p in points) - sum(p.x for p in points) * sum(p.y for p in points)) / \
    (n * sum(p.x**2 for p in points) - sum(p.x for p in points)**2)

b = (sum(p.y for p in points) / n) - m * sum(p.x for p in points) / n

# print(m, b) # 1.9393939393939394 4.7333333333333325

"""5-6. using inverse and transposed matrices to fit a linear regression"""
from numpy.linalg import inv
import numpy as np

# flatten input variables
X = X.flatten()

# add placeholder "1" column to generate intercept
X_1 = np.vstack([X, np.ones(len(X))]).transpose()

# calculate coefficients for slope and intercept
b = inv(X_1.transpose() @ X_1) @ (X_1.transpose() @ Y)
# print(b) # [1.93939394 4.73333333]

# predict against the y-values
y_predict = X_1.dot(b)

"""5-7. using QR decomposition to perform a linear regression"""
from numpy.linalg import qr

# calculate coefficients for slope and intercept
# using QR decomposition
Q, R = qr(X_1)
b = inv(R).dot(Q.transpose()).dot(Y)

# print(b) # [1.93939394 4.73333333]

"""5-8. using gradient descent to find the minimum of a parabola"""
import random

def f(x):
  return (x - 3)**2 + 4

def dx_f(x):
  return 2 * (x - 3)

learning_rate = 0.001

# number of iterations to perform gradient descent
iterations = 100_000 # often set to 1_000 iterations (Starmer, p. 95)

# start at a random x
x = random.randint(-15, 15)

for i in range(iterations):
    # get slope
    derivative = dx_f(x) 

    step_size = derivative * learning_rate

    # update x by subtracting the step size
    x -= step_size

# print(x, f(x)) # prints 2.999999999999889 4.0

"""5-9. performing gradient descent for a linear regression"""
# building the model
m = 0.0
b = 0.0

n = float(len(points)) # number of elements in X

# squared residual for a single observation/point
def get_sr(x, y):
  return (y - (m * x + b))**2

# derivative of squared residual with respect to m
def get_derivative_of_sr_wrt_m(x, y):
  return -2 * x * (y - (m * x + b))

# derivative of squared residual with respect to b
def get_derivative_of_sr_wrt_b(x, y):
  return -2 * (y - (m * x + b))

# perform gradient descent
for i in range(iterations):
    # slope of SSR with respect to m
    d_m = sum(get_derivative_of_sr_wrt_m(p.x, p.y) for p in points)

    # slope of SSR with respect to b
    d_b = sum(get_derivative_of_sr_wrt_b(p.x, p.y) for p in points)

    step_size_m = d_m * learning_rate
    step_size_b = d_b * learning_rate

    # update m and b by subtracting the step size
    m -= step_size_m
    b -= step_size_b

# print(f"y = {m}x + {b}")
# y = 1.9393939393939548x + 4.733333333333227

"""5-10. calculating partial derivatives for m and b using sympy"""
from sympy import *

m, b, i, n = symbols('m b i n')
x, y = symbols('x y', cls = Function)

# see ch.1 for reference on sympy usage
sum_of_squares = Sum((y(i) - (m * x(i) + b))**2, (i, 0, n))

d_m = diff(sum_of_squares, m)
d_b = diff(sum_of_squares, b)
# print(d_m)
# print(d_b)

# OUTPUTS
# Sum(-2*(-b - m*x(i) + y(i))*x(i), (i, 0, n))
# Sum(2*b + 2*m*x(i) - 2*y(i), (i, 0, n))

"""5-11. solving linear regression using sympy"""
# python indexes the data starting from 0
# sum_of_squares here mimics this by summing from 
# n=0 until 1 less than the total number of data points
d_m = diff(sum_of_squares, m) \
        .subs(n, len(points) - 1).doit() \
        .replace(x, lambda i: points[i].x) \
        .replace(y, lambda i: points[i].y)

d_b = diff(sum_of_squares, b) \
        .subs(n, len(points) - 1).doit() \
        .replace(x, lambda i: points[i].x) \
        .replace(y, lambda i: points[i].y)

# compile using lambdify for faster computation
# converts sympy to an optimized python function
d_m = lambdify([m, b], d_m)
d_b = lambdify([m, b], d_b)

# building the model
m = 0.0
b = 0.0

# perform gradient descent
for i in range(iterations):
    step_size_m = d_m(m, b) * learning_rate
    step_size_b = d_b(m, b) * learning_rate

    # update m and b by subtracting the step size
    m -= step_size_m
    b -= step_size_b

# print(f"y = {m}x + {b}")
# y = 1.939393939393954x + 4.733333333333231

"""5-12. plotting the loss function for linear regression"""
from sympy.plotting import plot3d

m, b, i, n = symbols('m b i n')
x, y = symbols('x y', cls = Function)

sum_of_squares = Sum((y(i) - (m * x(i) + b))**2, (i, 0, n)) \
                  .subs(n, len(points) - 1).doit() \
                  .replace(x, lambda i: points[i].x) \
                  .replace(y, lambda i: points[i].y)

# plot3d(sum_of_squares)

"""5-13. performing stochastic (or mini-batch) gradient descent for a linear regression"""
# input data
data = pd.read_csv('data/single_independent_variable_linear_small.csv', header=0)

X = data.iloc[:, 0].values
Y = data.iloc[:, 1].values

n = data.shape[0] # rows

# building the model
m = 0.0
b = 0.0

sample_size = 3 # set to 1 for stochastic gradient descent
learning_rate = 0.001
epochs = 1_000_000 # number of iterations to perform gradient descent

# performing stochastic (or mini-batch) gradient descent
"""
for i in range(epochs):
  # choose sample data point(s)
  idx = np.random.choice(n, sample_size, replace=False)
  x_sample = X[idx]
  y_sample = Y[idx]

  # MSE (mean squared error) as the loss or cost function
  # loss_function = sum((y_sample - m * x_sample + b)**2) / sample_size 

  # d/dm derivative of loss or cost function
  D_m = sum(-2 * x_sample * (y_sample - (m * x_sample + b))) / sample_size

  # d/db derivative of loss or cost function
  D_b = sum(-2 * (y_sample - (m * x_sample + b))) / sample_size

  step_size_m = D_m * learning_rate
  step_size_b = D_b * learning_rate

  # update m and b by subtracting the step size
  m -= step_size_m
  b -= step_size_b

  # print progress
  if i % 10_000 == 0:
    print(i, m, b)

print(f"y = {m}x + {b}")
# y = 1.940352620171648x + 4.763080622689704
"""

"""5-14. using pandas to see the correlation coefficient between every pair of variables"""
# print correlation between variables
correlations = df.corr(method='pearson')
# print(correlations)

# OUTPUT
#           x         y
# x  1.000000  0.957586
# y  0.957586  1.000000

"""5-15. calculating correlation coefficient from scratch in python"""
from math import sqrt

n = len(points)

numerator = n * sum(p.x * p.y for p in points) - \
            sum(p.x for p in points) * sum(p.y for p in points)

denominator = sqrt(n * sum(p.x**2 for p in points) - sum(p.x for p in points)**2) \
              * sqrt(n * sum(p.y**2 for p in points) - sum(p.y for p in points)**2)

corr = numerator / denominator

# print(corr) # 0.9575860952087218

"""5-16. calculating the critical value from a t-distribution"""
from scipy.stats import t

# t-distributions work with sample sizes of 30 or less
sample_size = 10
probability = .95
degrees_of_freedom = sample_size - 1
lower_tail_area, upper_tail_area = .025, .975

"""
https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.stats.t.html
"""
lower_critical_z = t(degrees_of_freedom).ppf(lower_tail_area)
upper_critical_z = t(degrees_of_freedom).ppf(upper_tail_area)

# print(lower_critical_z, upper_critical_z)
# -2.262157162740992 2.2621571627409915

"""5-17. testing significance for linear-looking data"""
# correlation coefficient derived from data
r = 0.957586

# perform the test
test_value = r / sqrt((1 - r**2) / (sample_size - 2))
# indicates strength of correlation with respect to SD
# will be negative if r is negative

"""
print(f"TEST VALUE: {test_value}")
print(f"CRITICAL RANGE: {lower_critical_z}, {upper_critical_z}")

if test_value < lower_critical_z or test_value > upper_critical_z:
  print("CORRELATION PROVEN, REJECT H0")
else:
  print("CORRELATION NOT PROVEN, FAILED to REJECT H0")
"""

# OUTPUT:
# TEST VALUE: 9.399564671312076
# CRITICAL RANGE: -2.262157162740992, 2.2621571627409915
# CORRELATION PROVEN, REJECT H0

# calculate p-value
if test_value > 0:
  p_value = 1.0 - t(degrees_of_freedom).cdf(test_value)
else:
  p_value = t(degrees_of_freedom).cdf(test_value)

# two-tailed, so multiply by 2
p_value = p_value * 2
# print(f"P-VALUE: {p_value}")

# OUTPUT:
# P-VALUE: 5.9763860877914965e-06

"""5-18. creating a correlation matrix in pandas"""
# print coefficient of determination (r**2) between variables
coeff_determination = df.corr(method='pearson')**2
# print(coeff_determination)

# OUTPUT
#           x         y
# x  1.000000  0.916971
# y  0.916971  1.000000

"""5-19. calculating the standard error of the estimate"""
# regression line
m = 1.939
b = 4.733

sample_size = 10
degrees_of_freedom = sample_size - 2 
# -2 because linear regression has two variables

# calculate standard error of estimate
standard_error = sqrt(
        (sum((p.y - (m*p.x + b))**2 for p in points)) / (degrees_of_freedom)
      )

# print(standard_error) # 1.87406793500129
# represents the standard deviation for a linear regression

"""5-20. calculating a prediction interval for x = 8.5"""
n = len(points)

# linear regression line
m = 1.939
b = 4.733

# calculate prediction interval for x = 8.5
x_0 = 8.5
x_mean = sum(p.x for p in points) / n

probability = .95
degrees_of_freedom = sample_size - 2
lower_tail_area, upper_tail_area = .025, .975

"""
https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.stats.t.html
"""
lower_critical_z = t(degrees_of_freedom).ppf(lower_tail_area)
upper_critical_z = t(degrees_of_freedom).ppf(upper_tail_area)

t_value = upper_critical_z

margin_of_error = t_value * standard_error * \
                  sqrt(1 + (1 / n) + (n * (x_0 - x_mean)**2) / \
                    (n * sum(p.x**2 for p in points) - \
                      sum(p.x for p in points)**2))

predicted_y = m*x_0 + b

# calculate prediction interval
# print(predicted_y - margin_of_error, predicted_y + margin_of_error)
# 16.462516875955465 25.966483124044537 

"""5-21. doing a train/test split on linear regression"""
from sklearn.model_selection import train_test_split

# extract input variables (all rows, all columns but last column)
X = df.values[:, :-1]

# extract output column (all rows, last column)
Y = df.values[:, -1]

# separate training and testing data
# this leaves a third of the data out for testing (1/3 is most common)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3)

model = LinearRegression()
model.fit(X_train, Y_train)
result = model.score(X_test, Y_test)
# print(f"r^2: {result:.3f}") # r^2: 0.850

"""5-22. using three-fold cross-validation for a linear regression"""
from sklearn.model_selection import KFold, cross_val_score

kfold = KFold(n_splits=3, random_state=7, shuffle=True)
model = LinearRegression()
results = cross_val_score(model, X, Y, cv=kfold)
# print(results)
# print(f"MSE: mean={results.mean():.3f} (stdev-{results.std():.3f})")

# OUTPUT:
# [0.82514286 0.64591573 0.2975653 ]
# MSE: mean=0.590 (stdev-0.219)

"""5-23. using a random-fold validation for a linear regression"""
from sklearn.model_selection import ShuffleSplit

# more computationally expensive 
# compared to train/test split or cross-validation
shufflesplit = ShuffleSplit(n_splits=10, test_size=.33, random_state=7)
model = LinearRegression()
results = cross_val_score(model, X, Y, cv=shufflesplit)

# print(results)
# print(f"MSE: mean={results.mean():.3f} (stdev-{results.std():.3f})")

# OUTPUT:
# [0.82514286 0.23552344 0.92653455 0.91620594 0.73260142 0.8698865
#  0.55254014 0.89593526 0.91570078 0.82086621]
# MSE: mean=0.769 (stdev-0.208)

"""5-24. a linear regression with two input variables"""
# load the data
df = pd.read_csv('data/multiple_independent_variable_linear.csv', delimiter=",")

# extract input variables (all rows, all columns but last column)
X = df.values[:, :-1]

# extract output column (all rows, last column)
Y = df.values[:, -1]

# training
fit = LinearRegression().fit(X, Y)

# print coefficients
# print(f"Coefficients = {fit.coef_}")
# print(f"Intercept = {fit.intercept_}")
# print(f"z = {fit.intercept_} + {fit.coef_[0]}x + {fit.coef_[1]}y")

# OUTPUT:
# Coefficients = [2.00672647 3.00203798]
# Intercept = 20.10943282003599
# z = 20.10943282003599 + 2.0067264725128062x + 3.002037976646691y

"""Exercises"""
# load the data
df = pd.read_csv('data/linear_normal.csv', delimiter=",")
# load the data with list() and .itertuples()
points = list(pd.read_csv('data/linear_normal.csv', delimiter=",").itertuples())

# 1. Perform a simple linear regression to find the m and b values that minimizes the loss (sum of squares).

# extract input variables (all rows, all columns but last column)
# .values method returns data points only (no index and header)
# use native accessor by tacking on [] 
X = df.values[:, :-1]

# extract output column (all rows, last column)
Y = df.values[:, -1]

# fit a line to the points
fit = LinearRegression().fit(X, Y)

m = fit.coef_.flatten()
b = fit.intercept_.flatten()
# print(f"m = {m}") # m = [1.75919315]
# print(f"b = {b}") # b = [4.69359655]

# show in chart
# plt.plot(X, Y, 'o')   # scatterplot
                        # 'o' - circle marker format
                        # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
# plt.plot(X, m*X + b)  # line
# plt.show()

# 2. Calculate the correlation coefficient and statistical significance of this data (at 95% confidence). Is the correlation useful?

"""(2A) using pandas to see the correlation coefficient between every pair of variables"""
# print correlation between variables
correlations = df.corr(method='pearson')
# print(correlations)

# OUTPUT
#          x        y
# x  1.00000  0.92421
# y  0.92421  1.00000

"""(2B) calculating the critical values at 95% confidence"""
sample_size = df.shape[0] # rows
degrees_of_freedom = sample_size - 1
lower_tail_area, upper_tail_area = .025, .975

"""
https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.stats.t.html
"""
lower_critical_z = t(degrees_of_freedom).ppf(lower_tail_area)
upper_critical_z = t(degrees_of_freedom).ppf(upper_tail_area)

# print(lower_critical_z, upper_critical_z)
# -1.9844674544266925 1.984467454426692

"""(2C) testing significance for linear-looking data"""
# correlation coefficient derived from data
r = correlations["y"]["x"]

# perform the test
test_value = r / sqrt((1 - r**2) / (sample_size - 2))
# indicates strength of correlation with respect to SD
# will be negative if r is negative

"""
print(f"TEST VALUE: {test_value}")
print(f"CRITICAL RANGE: {lower_critical_z}, {upper_critical_z}")

if test_value < lower_critical_z or test_value > upper_critical_z:
  print("CORRELATION PROVEN, REJECT H0")
else:
  print("CORRELATION NOT PROVEN, FAILED to REJECT H0")
"""

# OUTPUT:
# TEST VALUE: 23.83551532367729
# CRITICAL RANGE: -1.9844674544266925, 1.984467454426692
# CORRELATION PROVEN, REJECT H0

# calculate p-value
if test_value > 0:
  p_value = 1.0 - t(degrees_of_freedom).cdf(test_value)
else:
  p_value = t(degrees_of_freedom).cdf(test_value)

# two-tailed, so multiply by 2
p_value = p_value * 2
# print(f"P-VALUE: {p_value}")

# OUTPUT:
# P-VALUE: 0.0 (extremely small)

# 3. If I predict where x = 50, what is the 95% prediction interval for the predicted value of y?

"""(3A) calculating the standard error of the estimate"""
# linear regression line
m = 1.75919315
b = 4.69359655

degrees_of_freedom = sample_size - 2 
# -2 because linear regression has two variables

# calculate standard error of estimate
standard_error = sqrt(
        (sum((p.y - (m*p.x + b))**2 for p in points)) / (degrees_of_freedom)
      )

# print(standard_error) # 20.98596726693759
# represents the standard deviation for a linear regression

"""(3B) calculating a prediction interval for x = 50"""
n = len(points)

# calculate prediction interval for x = 50
x_0 = 50
x_mean = sum(p.x for p in points) / n

t_value = upper_critical_z

margin_of_error = t_value * standard_error * \
                  sqrt(1 + (1 / n) + (n * (x_0 - x_mean)**2) / \
                    (n * sum(p.x**2 for p in points) - \
                      sum(p.x for p in points)**2))

predicted_y = m*x_0 + b

# calculate prediction interval
# print(predicted_y - margin_of_error, predicted_y + margin_of_error)
# 50.797480310779555 134.50902778922045 

# 4. Start your regression over and do a train/test split. Feel free to experiment with cross-validation and random-fold validation. Does the linear regression perform well and consistently on the testing data? Why or why not?
kfold = KFold(n_splits=3, random_state=7, shuffle=True)
model = LinearRegression()
results = cross_val_score(model, X, Y, cv=kfold)
# print(results)
# print(f"MSE: mean={results.mean():.3f} (stdev-{results.std():.3f})")

# OUTPUT:
# [0.86119665 0.78237719 0.85733887]
# MSE: mean=0.834 (stdev-0.036)
# the linear regression performed moderately well
