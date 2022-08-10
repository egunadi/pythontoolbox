"""2-1. using bayes' theorem in python"""
p_coffee_drinker = .65
p_cancer = .005
p_coffee_drinker_given_cancer = .85

p_cancer_given_coffee_drinker = (p_coffee_drinker_given_cancer * p_cancer) / p_coffee_drinker

# print(p_cancer_given_coffee_drinker) # prints 0.006538461538461539

"""2-2. using scipy for the binomial distribution"""
from scipy.stats import binom

n = 10  # number of trials 
p = 0.9 # underlying probability of success 

for k in range(n + 1):
  probability = binom.pmf(k, n, p)
  # print("{0} - {1}".format(k, probability))
  # print(f"{k} - {probability}") # using an f-string

"""2-3. beta distribution using scipy"""
from scipy.stats import beta

a = 8 # number of successes
b = 2 # number of failures
p = beta.cdf(.90, a, b) # cdf function calculates AUC up to a point
# print(p) # 0.7748409780000001

"""2-4. subtracting to get a right area in a beta distribution"""
p = 1.0 - beta.cdf(.90, a, b)
# print(p) # 0.22515902199999993

"""2-6. beta distribution middle area using scipy"""
p = beta.cdf(.90, a, b) - beta.cdf(.80, a, b)
# print(p) # 0.33863336200000016

"""Exercises"""
# 1. There is a 30% chance of rain today, and a 40% chance your umbrella order will arrive on time. You are eager to walk in the rain today and cannot do so without either! 
# What is the probability it will rain AND your umbrella will arrive?
p_rain = 0.3
p_umbrellaArrives = 0.4
p_rain_and_umbrellaArrives = p_rain * p_umbrellaArrives
# print(p_rain_and_umbrellaArrives) # 0.12

# 2. There is a 30% chance of rain today, and a 40% chance your umbrella order will arrive on time.
# You will be able to run errands only if it does not rain or your umbrella arrives.
# What is the probability it will not rain OR your umbrella arrives?
p_notRain = 1 - p_rain
p_notRain_and_umbrellaArrives = p_notRain * p_umbrellaArrives
p_notRain_or_umbrellaArrives = p_notRain + p_umbrellaArrives - p_notRain_and_umbrellaArrives
# print(p_notRain_or_umbrellaArrives) # 0.8200000000000001

# 3. There is a 30% chance of rain today, and a 40% chance your umbrella order will arrive on time.
# However, you found out if it rains there is only a 20% chance your umbrella will arrive on time.
# What is the probability it will rain AND your umbrella will arrive on time?
p_umbrellaArrives_given_rain = 0.2
p_rain_and_umbrellaArrives = p_rain * p_umbrellaArrives_given_rain
# print(p_rain_and_umbrellaArrives) # 0.06

# 4. You have 137 passengers booked on a flight from Las Vegas to Dallas. However, it is Las Vegas on a Sunday morning and you estimate each passenger is 40% likely to not show up.
# You are trying to figure out how many seats to overbook so the plane does not fly empty.
# How likely is it at least 50 passengers will not show up?
passenger_count = 137 # number of trials
p_missing = 0.4       # underlying probability of success,
                      # defined as a missing passenger

# see a table of probabilities associated with number of missing passengers
for k in range(passenger_count + 1):
  probability = binom.pmf(k, passenger_count, p_missing)
  # print(f"{k} - {probability}") # using an f-string

p_atLeast50_missing = sum(binom.pmf(k, passenger_count, p_missing) 
                          for k in range(50, passenger_count + 1))

# print(p_atLeast50_missing) # 0.8220955881474781

# 5. You flipped a coin 19 times and got heads 15 times and tails 4 times.
# Do you think this coin has any good probability of being fair? Why or why not?

heads = 15 # number of successes, defined as number of heads
tails = 4 # number of failures, defined as number of tails

# coin seems to be rigged for heads -- let's verify
p = 1.0 - beta.cdf(0.5, heads, tails) 
print(p) # 0.9962310791015625 probability that underlying probability for heads is over 0.5
