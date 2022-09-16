"""6-1. the logistic function in python for one independent variable"""
import math

def get_logistic_function_probability(
  x:  float,
  b0: float,
  b1: float
) ->  float:
  p = 1.0 / (1.0 + math.exp(-(b0 + b1 * x)))
  return p

"""6-2. using sympy to plot a logistic function"""
from sympy import *

b0, b1, x = symbols('b0 b1 x')

p = 1.0 / (1.0 + exp(-(b0 + b1 * x)))

p = p.subs(b0, -2.823)
p = p.subs(b1, 0.620)

# print(p) # 1.0/(1.0 + 16.8272567955368*exp(-0.62*x))
# plot(p)

"""6-3. using a plain logistic regression in scipy"""
import pandas as pd
from sklearn.linear_model import LogisticRegression

# load the data
df = pd.read_csv('data/simple_logistic_regression.csv', delimiter=",")

# extract input variables (all rows, all columns but last column)
# .values method returns data points only (no index and header)
# use native accessor by tacking on [] 
X = df.values[:, :-1]

# extract output column (all rows, last column)
Y = df.values[:, -1]

# perform logistic regression
# turn off penalty
model = LogisticRegression(penalty='none')
model.fit(X, Y)

# print beta1
# print(model.coef_.flatten()) # [0.69267212]

# print beta0
# print(model.intercept_.flatten()) # [-3.17576395]


"""6-4. calculating the joint likelihood of observing all the points for a given logistic regression"""
# import points with .itertuples()
points = pd.read_csv('data/simple_logistic_regression.csv', delimiter=",").itertuples()

b0 = -3.17576395
b1 = 0.69267212

# calculate the joint likelihood
joint_likelihood = 1.0

for p in points:
  if p.y == 1.0:
    joint_likelihood *= get_logistic_function_probability(p.x, b0, b1)
  elif p.y == 0.0:
    joint_likelihood *= (1.0 - get_logistic_function_probability(p.x, b0, b1))

# print(joint_likelihood) # 4.7911180221699105e-05

"""6-5. compressing the joint likelihood calculation without an `if` expression"""
for p in points:
  joint_likelihood *= get_logistic_function_probability(p.x, b0, b1)**p.y \
                        * (1.0 - get_logistic_function_probability(p.x, b0, b1))**(1.0 - p.y)

"""6-6. using logarithmic addition"""
joint_likelihood = 0.0

for p in points:
  joint_likelihood += math.log(
                        get_logistic_function_probability(p.x, b0, b1)**p.y \
                          * (1.0 - get_logistic_function_probability(p.x, b0, b1))**(1.0 - p.y)
                      )
 
joint_likelihood = math.exp(joint_likelihood)

"""6-7. expressing a joint likelihood for logistic regression in sympy"""
b0, b1, i, n = symbols('b0 b1 i n')
x, y = symbols('x y', cls=Function)

joint_likelihood =  Sum(
                      log(
                        (1.0 / (1.0 + exp(-(b0 + b1 * x(i)))))**y(i) \
                          * (1.0 - (1.0 / (1.0 + exp(-(b0 + b1 * x(i))))))**(1 - y(i))
                      ), 
                      (i, 0, n)
                    )

"""6-8. using gradient descent on logistic regression"""
# load the data with list() and .itertuples()
points = list(pd.read_csv('data/simple_logistic_regression.csv', delimiter=",").itertuples())

# partial derivative for b1, with points substituted
d_b1 = diff(joint_likelihood, b1) \
        .subs(n, len(points) - 1).doit() \
        .replace(x, lambda i: points[i].x) \
        .replace(y, lambda i: points[i].y)

# partial derivative for b0, with points substituted
d_b0 = diff(joint_likelihood, b0) \
        .subs(n, len(points) - 1).doit() \
        .replace(x, lambda i: points[i].x) \
        .replace(y, lambda i: points[i].y)

# compile using lambdify for faster computation
d_b1 = lambdify([b1, b0], d_b1)
d_b0 = lambdify([b1, b0], d_b0)

# perform gradient descent
b1 = 0.01
b0 = 0.01
learning_rate = .01

# number of iterations to perform gradient descent
iterations = 10_000 # often set to 1_000 iterations (Starmer, p. 95)

for j in range(iterations):
  # slope of joint likelihood function with respect to b1
  derivative_wrt_b1 = d_b1(b1, b0)

  # slope of joint likelihood function with respect to b0
  derivative_wrt_b0 = d_b0(b1, b0)

  step_size_b1 = derivative_wrt_b1 * learning_rate
  step_size_b0 = derivative_wrt_b0 * learning_rate

  # update b1 and b0 by adding the step size
  # we're not substracting since we're finding a local maximum, not minimum
  # the local maximum will hopefully lead to the maxmimum likelihood function
  b1 += step_size_b1
  b0 += step_size_b0

# print(b1, b0) # 0.6926693075370812 -3.175751550409821

"""6-9. doing a multivariable logistic regression on employee data"""
# load the data
employee_data = pd.read_csv("data/employee_retention_analysis.csv")

# grab independent variable columns (all rows, all columns but last column)
inputs = employee_data.iloc[:, :-1]

# grab dependent "did_quit" variable column (all rows, last column)
output = employee_data.iloc[:, -1]

# build logistic regression
# turn off penalty
model = LogisticRegression(penalty='none')
fit = model.fit(inputs, output)

# print coefficients:
# print(f"COEFFICIENTS: {fit.coef_.flatten()}")
# print(f"INTERCEPT: {fit.intercept_.flatten()}")

# OUTPUT
# COEFFICIENTS: [ 0.03213405  0.03682453 -2.50410028  0.9742266 ]
# INTERCEPT: [-2.73485301]

# interact and test with new employee data
def predict_employee_will_stay(
  sex:            int,
  age:            int,
  promotions:     int,
  years_employed: int
) ->              str:
  prediction = fit.predict([[sex, age, promotions, years_employed]])
  probabilities = fit.predict_proba([[sex, age, promotions, years_employed]])

  if prediction == [[1]]:
    return f"WILL LEAVE: {probabilities}"
  else:
    return f"WILL STAY: {probabilities}"

# test a prediction
# while True:
#   user_input = input("Predict employee will stay or leave {sex}, {age}, {promotions}, {years_employed}: ")

#   (sex, age, promotions, years_employed) = user_input.split(",")

#   print(
#     predict_employee_will_stay(
#       int(sex),
#       int(age),
#       int(promotions),
#       int(years_employed)
#     )
#   )

# OUTPUT using [1,34,1,5]:
# WILL LEAVE: [[0.28570264 0.71429736]] 
# first output is probability of false (0)
# second output is probability of true (1)

"""6-10. calculating the log likelihood of the fit"""
from math import log, exp

# import points with .itertuples()
points = pd.read_csv('data/simple_logistic_regression.csv', delimiter=",").itertuples()

# declare fitted logistic regression
b0 = -3.17576395
b1 = 0.69267212

# sum the log-likelihoods
log_likelihood_fit = 0.0

for p in points:
  if p.y == 1.0:
    log_likelihood_fit += log(get_logistic_function_probability(p.x, b0, b1))
  elif p.y == 0.0:
    log_likelihood_fit += log(1.0 - get_logistic_function_probability(p.x, b0, b1))

# print(log_likelihood_fit) # -9.946161673231583

"""6-11. consolidating our log likelihood logic into a single line (without an `if` expression)"""
# load the data with list() and .itertuples()
points = list(pd.read_csv('data/simple_logistic_regression.csv', delimiter=",").itertuples())

log_likelihood_fit =  sum(
                        log(get_logistic_function_probability(p.x, b0, b1)) * p.y \
                          + log(1.0 - get_logistic_function_probability(p.x, b0, b1)) * (1.0 - p.y)
                        for p in points
                      )

"""6-12. log likelihood of patients"""
likelihood = sum(p.y for p in points) / len(points)

log_likelihood = 0.0

for p in points:
  if p.y == 1.0:
    log_likelihood += log(likelihood)
  elif p.y == 0.0:
    log_likelihood += log(1.0 - likelihood)

# print(log_likelihood) # -14.341070198709906

"""6-13. consolidating the log likelihood into a single line"""
log_likelihood =  sum(
                    log(likelihood) * p.y \
                      + log(1.0 - likelihood) * (1.0 - p.y)
                    for p in points
                  )

"""6-14. calculating the R-square for a logistic regression"""
r2 = (log_likelihood - log_likelihood_fit) / log_likelihood

# print(r2) # 0.306456105756576

"""6-15. calculating p-value for a given logistic regression"""
from scipy.stats import chi2

chi2_input = 2 * (log_likelihood_fit - log_likelihood)
p_value = chi2.pdf(chi2_input, 1) # 1 degree of freedom (n - 1)

# print(p_value) # 0.0016604875618753787

"""6-16. performing a logistic regression with three-fold cross-validation"""
from sklearn.model_selection import KFold, cross_val_score

# grab independent variable columns (all rows, all columns but last column)
# .values method returns data points only (no index and header)
X = employee_data.values[:, :-1]

# grab dependent "did_quit" variable column (all rows, last column)
Y = employee_data.values[:, -1]

# "random_state" is the random seed, which we fix to 7
kfold = KFold(n_splits=3, random_state=7, shuffle=True)
model = LogisticRegression(penalty='none')
results = cross_val_score(model, X, Y, cv=kfold)

# print(f"Accuracy Mean: {results.mean():.3f} (stdev={results.std():.3f})")

# OUTPUT
# Accuracy Mean: 0.611 (stdev=0.000)

"""6-17. creating a confusion matrix for a testing dataset in scipy"""
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

model = LogisticRegression(solver='liblinear')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.33, random_state=10)
model.fit(X_train, Y_train)
prediction = model.predict(X_test)

"""
The confusion matrix evaluates accuracy within each category.
[[truepositives falsenegatives
 [falsepositives truenegatives]]

The diagnonal represents correct predictions,
so we want those to be higher
"""
matrix = confusion_matrix(y_true=Y_test, y_pred=prediction)
# print(matrix)

# OUTPUT
# [[6 3]
#  [4 5]]

"""6-18. using the AUC as the scikit-learn parameter"""
results = cross_val_score(model, X, Y, cv=kfold, scoring='roc_auc')
# print(f"AUC: {results.mean():.3f} ({results.std():.3f})")

# OUTPUT
# AUC: 0.791 (0.051)

"""6-19. using the `stratify` option in scikit-learn to balance classes in the data"""
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.33, stratify=Y)

"""Exercises"""
# A dataset of three input variables RED, GREEN, and BLUE as well as an output variable LIGHT_OR_DARK_FONT_IND is provided here. It will be used to predict whether a light/dark font (0/1 respectively) will work for a given background color (specified by RGB values).
rgb_data = pd.read_csv("data/light_dark_font_training_set.csv")

# 1. Perform a logistic regression on the preceding data, using three-fold cross-validation and accuracy as your metric.

# grab independent variable columns (all rows, all columns but last column)
# .values method returns data points only (no index and header)
X = rgb_data.values[:, :-1]

# grab dependent "LIGHT_OR_DARK_FONT_IND" variable column (all rows, last column)
Y = rgb_data.values[:, -1]

# "random_state" is the random seed, which we fix to 7
kfold = KFold(n_splits=3, random_state=7, shuffle=True)
model = LogisticRegression(penalty='none')
results = cross_val_score(model, X, Y, cv=kfold)

# print(f"Accuracy Mean: {results.mean():.3f} (stdev={results.std():.3f})")

# OUTPUT
# Accuracy Mean: 1.000 (stdev=0.000)

# 2. Produce a confusion matrix comparing the predictions and actual data.
model = LogisticRegression(solver='liblinear')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.33, random_state=10)
model.fit(X_train, Y_train)
prediction = model.predict(X_test)

"""
The confusion matrix evaluates accuracy within each category.
[[truepositives falsenegatives
 [falsepositives truenegatives]]

The diagnonal represents correct predictions,
so we want those to be higher
"""
matrix = confusion_matrix(y_true=Y_test, y_pred=prediction)
# print(matrix)

# OUTPUT
# [[169   5]
#  [  2 268]]

# 3. Pick a few different background colors (you can use an RGB tool like this one) and see if the logistic regression sensibly chooses a light (0) or dark (1) font for each one.

# build logistic regression
# turn off penalty
model = LogisticRegression(penalty='none')
fit = model.fit(X_train, Y_train)

# print coefficients:
# print(f"COEFFICIENTS: {fit.coef_.flatten()}")
# print(f"INTERCEPT: {fit.intercept_.flatten()}")

# OUTPUT
# COEFFICIENTS: [ 0.03213405  0.03682453 -2.50410028  0.9742266 ]
# INTERCEPT: [-2.73485301]

# interact and test with new rgb data
def predict_light_or_dark_font(
  r:  int,
  g:  int,
  b:  int
) ->  str:
  prediction = fit.predict([[r, g, b]])
  probabilities = fit.predict_proba([[r, g, b]])

  if prediction == [[1]]:
    return "DARK"
  else:
    return "LIGHT"

# test a prediction
# while True:
#   user_input = input("Predict font color will be light or dark {r}, {g}, {b}: ")

#   (r, g, b) = user_input.split(",")

#   print(
#     predict_light_or_dark_font(
#       int(r),
#       int(g),
#       int(b)
#     )
#   )

# OUTPUT using [0,0,0]:
# LIGHT 

# OUTPUT using [255,255,255]:
# DARK

# 4. Based on the preceding exercises, do you think logistic regression is effective for predicting a light or dark font for a given background color?
# Yes, the Accuracy Mean is very high. Also, the confusion matrix has high numbers in the top-left to bottom-right diagonal with low numbers in the other cells.