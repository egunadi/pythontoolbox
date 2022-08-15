"""4-4. declaring a five-dimensional vector using numpy"""
import numpy as np

v = np.array([6, 1, 5, 8, 3])
# print(v)

"""4-5. adding two vectors using numpy"""
from numpy import array

v = array([3, 2])
w = array([2, -1])
v_plus_w = v + w
# print(v_plus_w) # [5 1]


"""4-6. scaling a vector using numpy"""
v = array([3, 1])
scaled_v = 2.0 * v
# print(scaled_v) # [6. 2.]

"""4-7. matrix vector multiplication in numpy"""
# compose basis matrix with i-hat and j-hat
basis = array(
  [[3, 0],
   [0, 2]]
)

# declare vector v
v = array([1, 1])

# create new vector
# by transforming v with dot product
new_v = basis.dot(v)

# print(new_v) # [3 2]

"""4-8. separating the basis vectors and applying them as a transformation"""
# declare i-hat and j-hat
i_hat = array([2, 0])
j_hat = array([0, 3])

# compose basis matrix using i-hat and j-hat
# also need to transpose rows into columns
basis = array([i_hat, j_hat]).transpose()

# declare vector v
v = array([1, 1])

# create new vector
# by transforming v with dot product
new_v = basis.dot(v)

# print(new_v) # [2 3]

"""4-10. a more complicated transformation"""
# declare i-hat and j-hat
i_hat = array([2, 3])
j_hat = array([2, -1])

# compose basis matrix using i-hat and j-hat
# also need to transpose rows into columns
basis = array([i_hat, j_hat]).transpose()

# declare vector v
v = array([2, 1])

# create new vector
# by transforming v with dot product
new_v = basis.dot(v)

# print(new_v) # [6 5]

"""4-11. combining two transformations"""
# transformation 1
i_hat1 = array([0, 1])
j_hat1 = array([-1, 0])
transform1 = array([i_hat1, j_hat1]).transpose()

# transformation 2
i_hat2 = array([1, 0])
j_hat2 = array([1, 1])
transform2 = array([i_hat2, j_hat2]).transpose()

# combine transformations using `matmul()` or `@` operator
# note that the order you apply each transformation matters!
combined_transform = transform2 @ transform1

# test
# print(f"COMBINED MATRIX:\n {combined_transform}")

v = array([1, 2])
# print(combined_transform.dot(v)) # [-1  1]
# or
rotated = transform1.dot(v)
sheered = transform2.dot(rotated)
# print(sheered) # [-1  1]

"""4-13. calculating a determinant"""
from numpy.linalg import det

i_hat = array([3, 0])
j_hat = array([0, 2])

basis = array([i_hat, j_hat]).transpose()
determinant = det(basis)
# print(determinant) # prints 6.0

"""4-17. using sympy to study the inverse and identity matrix"""
from sympy import *

# 4x + 2y + 4z = 44
# 5x + 3y + 7z = 56
# 9x + 3y + 6z = 72
A = Matrix([
  [4, 2, 4],
  [5, 3, 7],
  [9, 3, 6]
])

# dot product between A and its inverse
# will produce identity function
inverse = A.inv()
identity = inverse * A

# prints Matrix([[-1/2, 0, 1/3], [11/2, -2, -4/3], [-2, 1, 1/3]])
# print(f"INVERSE: {inverse}")
# prints Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# print(f"IDENTITY: {identity}")

"""4-18. using numpy to solve a system of equations"""
from numpy.linalg import inv

A = array([
  [4, 2, 4],
  [5, 3, 7],
  [9, 3, 6]
])

B = array([
  44,
  56,
  72
])

X = inv(A).dot(B)
# print(X) # [ 2. 34. -8.]

"""4-19. using sympy to solve a system of equations"""
A = Matrix([
  [4, 2, 4],
  [5, 3, 7],
  [9, 3, 6]
])

B = Matrix([
  44,
  56,
  72
])

X = A.inv() * B
# print(X) # Matrix([[2], [34], [-8]])

"""4-20. performing eigendecomposition in numpy"""
from numpy import diag
from numpy.linalg import eig

# eigendecomposition only works on square matrices
A = array([
  [1, 2],
  [4, 5]
])

eigenvals, eigenvecs = eig(A)
# print("EIGENVALUES")
# print(eigenvals)
# print("\nEIGENVECTORS")
# print(eigenvecs)

"""
EIGENVALUES
[-0.46410162  6.46410162]

EIGENVECTORS
[[-0.80689822 -0.34372377]
 [ 0.59069049 -0.9390708 ]]
"""

"""4-21. recomposing a matrix in numpy"""
Q = eigenvecs
R = inv(Q)

L = diag(eigenvals)
B = Q @ L @ R

# print("REBUILD MATRIX")
# print(B)

"""
REBUILD MATRIX
[[1. 2.]
 [4. 5.]]
"""

"""Exercises"""
# 1. Vector v has a value of [1, 2] but then a transformation happens. i_hat lands at [2, 0] and j_hat lands at [0, 1.5]. Where does v land?
i_hat = array([2, 0])
j_hat = array([0, 1.5])

# compose basis matrix using i-hat and j-hat
# also need to transpose rows into columns
basis = array([i_hat, j_hat]).transpose()

# declare vector v
v = array([1, 2])

# create new vector
# by transforming v with dot product
new_v = basis.dot(v)

# print(new_v) # [2. 3.]

# 2. Vector v has a value of [1, 2] but then a transformation happens. i_hat lands at [-2, 1] and j_hat lands at [1, -2]. Where does v land?
i_hat = array([-2, 1])
j_hat = array([1, -2])

# compose basis matrix using i-hat and j-hat
# also need to transpose rows into columns
basis = array([i_hat, j_hat]).transpose()

# declare vector v
v = array([1, 2])

# create new vector
# by transforming v with dot product
new_v = basis.dot(v)

# print(new_v) # [ 0 -3]

# 3. A transformation i_hat lands at [1, 0] and j_hat lands at [2, 2]. What is the determinant of this transformation?
from numpy.linalg import det

i_hat = array([1, 0])
j_hat = array([2, 2])

basis = array([i_hat, j_hat]).transpose()
determinant = det(basis)
# print(determinant) # 2.0

# 4. Can two or more linear transformations be done in single linear transformation? Why or why not?
# yes, we can combine their transformations by multiplying their respective matrices into one matrix 

# 5. Solve the system of equations for x, y, and z:
# 3x + 1y + 0z = 54
# 2x + 4y + 1z = 12
# 3x + 1y + 8z = 6

# A = Matrix([
#   [3, 1, 0],
#   [2, 4, 1],
#   [3, 1, 8]
# ])
# or
A = array([
  [3, 1, 0],
  [2, 4, 1],
  [3, 1, 8]
])

# B = Matrix([
#   54,
#   12,
#   6
# ])
# or 
B = array([
  54,
  12,
  6
])

# X = A.inv() * B
# or
X = inv(A).dot(B)

# print(X) 
"""
Matrix([[99/5], [-27/5], [-6]])
or
[19.8 -5.4 -6. ]
"""

# 6. Is the following matrix linearly dependent? Why or why not?
# Matrix([[2, 1], [6, 3])
i_hat = array([2, 1])
j_hat = array([6, 3])

# basis = array([i_hat, j_hat]).transpose()
# or
from sympy import *
basis = Matrix([
  [2, 1],
  [6, 3]
])

determinant = det(basis)
print(determinant) # 0 determinant, meaning the matrix is linearly dependent
