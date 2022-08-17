from sympy import *
from sympy.plotting import plot3d

"""1-7. linear function using sympy"""
x = symbols('x')
f = 2*x + 1
# plot(f)

"""1-8. charting an exponential function"""
x, y = symbols('x y')
f = 2*x + 3*y
# plot3d(f)

"""
1-10. performing a summation
note that range() is end exclusive
"""
summation = sum(2*i for i in range(1, 6))
# print(summation) # 30

"""1-11. summation of elements"""
x = [1, 4, 6, 2]
n = len(x)
summation = sum(10*x[i] for i in range(0, n))
# print(summation) # 130

"""with sympy"""
i, n = symbols('i n')
# https://docs.sympy.org/latest/modules/concrete.html
summation = Sum(2*i, (i, 1, n)) 
up_to_5 = summation.subs(n, 5) # Sum() is end inclusive
# print(up_to_5.doit()) # 30

"""simplify expressions with sympy"""
x = symbols('x')
expr = x**2 / x**5
# print(expr) # x**(-3)

"""1-12. using log()"""
from math import log
x = log(8, 2)
# print(x) # 3.0

"""1-13. calculating compound interest in python"""
from math import exp
p = 100 # principal, starting amount
r = .20 # interest rate, by year
t = 2.0 # time, number of years
n = 12

a = p * (1 + (r/n))**(n * t)
# print(a) # prints 148.69146179463576

"""1-14. calculating continuous interest"""
a = p * exp(r*t) # no need for "n"
# print(a) # prints 149.18246976412703

"""1-16. using sympy to calculate limits"""
x = symbols('x')
f = 1 / x
result = limit(f, x, oo)
# print(result) # 0

n = symbols('n')
f = (1 + (1/n))**n
result = limit(f, n, oo)
# print(result) # E
# print(result.evalf()) # 2.71828182845905

"""1-17. a derivative calculator in python"""
def get_derivative_x(f, x, step_size):
  m = (f(x + step_size) - f(x)) / ((x + step_size) - x)
  return m

def my_function(x):
  return x**2

slope_at_2 = get_derivative_x(my_function, 2, .00001)
# print(slope_at_2) # prints 4.000010000000827

"""1-18. calculating a derivative in sympy"""
x = symbols('x')
f = x**2
dx_f = diff(f)
# print(dx_f) # prints 2*x
# print(dx_f.subs(x,2)) # prints 4

# alternatively, take it back to python...
def f(x):
  return x**2

def dx_f(x):
  return 2*x

slope_at_2 = dx_f(2.0)
# print(slope_at_2) # prints 4.0

"""1-21. calculating partial derivatives with sympy"""
x, y = symbols('x y')
f = 2*x**3 + 3*y**3
dx_f = diff(f, x)
dy_f = diff(f, y)
# print(dx_f) # prints 6*x**2
# print(dy_f) # prints 9*y**2
# plot3d(f)

"""1-22. using limits to calculate a slope"""
x, s = symbols('x s') # "s" for step size
f = x**2
slope_f = (f.subs(x, x+s) - f) / ((x+s) - x)
result = limit(slope_f, s, 0)
# print(result) # 2x

slope_2 = slope_f.subs(x, 2)
result = limit(slope_2, s, 0)
# print(result) # 4

"""
1-25. calculating the derivative dz/dx
with and without the chain rule, 
but still getting the same answer
"""
x, y = symbols('x y')

_y = x**2 + 1 # using _y to prevent variable clash
dy_dx = diff(_y)

z = y**3 - 2
dz_dy = diff(z)

dz_dx_chain = (dy_dx * dz_dy).subs(y, _y)
dz_dx_no_chain = diff(z.subs(y, _y))

# print(dz_dx_chain) # 6*x*(x**2 + 1)**2
# print(dz_dx_no_chain) # 6*x*(x**2 + 1)**2

"""1-26. integral approximation in python"""
def approximate_integral(a, b, n, f):
  delta_x = (b - a) / n # width of each rectangle
  total_sum = 0
  for i in range(1, n + 1): # range() is end exclusive
    midpoint = 0.5 * (2*a + delta_x*(2*i - 1))
    total_sum += f(midpoint)
  return total_sum * delta_x

def my_function(x):
  return x**2 + 1

area = approximate_integral(a=0, b=1, n=5, f=my_function)
# print(area) # prints 1.33

area = approximate_integral(a=0, b=1, n=1_000_000, f=my_function)
# print(area) # prints 1.3333333333332733

"""1-29. using sympy to perform integration"""
x = symbols('x')
f = x**2 + 1
area = integrate(f, (x, 0, 1))
# print(area) # prints 4/3

"""1-30. using limits to calculate integrals"""
x, i, n = symbols('x i n')
f = x**2 + 1
lower, upper = 0, 1
delta_x = ((upper - lower) / n) # width of each rectangle
x_i = (lower + delta_x * i)
fx_i = f.subs(x, x_i)
n_rectangles = Sum(delta_x * fx_i, (i, 1, n)).doit()
area = limit(n_rectangles, n, oo)
# print(area) # prints 4/3

"""Exercises"""
# 1. Is the value 62.6738 rational or irrational? Why or why not?
# Rational because it has a finite number of decimals, and therefore can be expressed as a fraction 626_738 / 10_000

# 2. Evaluate the expression: 10**7 * 10**-5
f = x**7 * x**-5
# print(f)
# print(f.subs(x, 10))
# 10**2 = 100

# 3. Evaluate the expression: 81**(1/2)
# 9

# 4. Evaluate the expression: 25**(3/2)
# 125

# 5. Assuming no payments are made, how much would a $1,000 loan be worth at 5% interest compounded monthly after 3 years?
p = 1_000 # principal, starting amount
r = .05 # interest rate, by year
t = 3.0 # time, number of years
n = 12

a = p * (1 + (r/n))**(n * t)
# print(a) # prints 1161.4722313334678

# 6. Assuming no payments are made, how much would a $1,000 loan be worth at 5% interest compounded continuously after 3 years?
a = p * exp(r*t) # no need for "n"
# print(a) # prints 1161.834242728283)

# 7. For the function f(x) = 3*x**2 + 1 what is the slope at x = 3?
x = symbols('x')
f = 3*x**2 + 1
dx_f = diff(f)
# print(dx_f) # prints 6*x
# print(dx_f.subs(x,3)) # prints 18

# 8. For the function f(x) = 3*x**2 + 1 what is the area under the curve for x between 0 and 2?
x = symbols('x')
f = 3*x**2 + 1
area = integrate(f, (x, 0, 2))
# print(area) # prints 10
