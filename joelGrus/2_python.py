"""2-1. object-oriented programming"""

# define a class using PascalCase name
class CountingClicker:
    """A class can/should have a docstring, just like a function"""

    # "dunder"/"magic" methods with "special" behaviors
    def __init__(self, count = 0):
      self.count = count

    def __repr__(self):
      return f"CountingClicker(count={self.count})"

    # public API of our class
    def click(self, num_times = 1):
      """Click the clicker some number of times."""
      self.count += num_times

    def read(self):
      return self.count

    def reset(self):
      self.count = 0

# test cases
clicker = CountingClicker()
assert clicker.read() == 0, "clicker should start with count 0"

clicker.click()
clicker.click()
assert clicker.read() == 2, "after two clicks, clicker should have count 2"

clicker.reset()
assert clicker.read() == 0, "after reset, clicker should be back to 0"

# a subclass inherits all the behavior of its parent class
class NoResetClicker(CountingClicker):
  # this class has all the same methods as CountingClicker

  # except that it has a reset method that does nothing
  def reset(self):
    pass

# test cases
clicker2 = NoResetClicker()
assert clicker2.read() == 0, "clicker should start with count 0"

clicker2.click()
assert clicker2.read() == 1, "after one clicks, clicker should have count 1"

clicker2.reset()
assert clicker2.read() == 1, "reset shouldn't do anything"

"""2-2. iterables and generators"""
# mimic range(), which is itself lazy
def generate_range(n):
  i = 0
  while i < n: # range is end exclusive
    yield i # every call to yield produces a value of the generator
    i += 1

# for i in generate_range(10):
#   print(f"i: {i}")

# create an infinite sequence
def generate_natural_numbers():
  """returns 1, 2, 3, ..."""
  n = 1
  while True:
    yield n
    n += 1

# none of these computations do anything until we iterate
data = generate_natural_numbers()

evens = ( number 
          for number in data 
          if number % 2 == 0 )

even_squares = (number**2
                for number in evens)

even_squares_ending_in_six = (number
                              for number in even_squares
                              if number % 10 == 6)

# turn list/generator values into (index, value) pairs with enumerate
names_list = ["Alice", "Bob", "Charlie", "Debbie"]

# for i, name in enumerate(names_list):
#   print(f"name {i} is {name}")
