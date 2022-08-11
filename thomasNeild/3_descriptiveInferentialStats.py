"""3-1. calculating mean in python"""
sample = [1, 3, 2, 5, 7, 0, 2, 3]
mean = sum(sample) / len(sample)
# print(mean) # prints 2.875

"""3-2. calculating a weighted mean in python"""
sample  = [90, 80, 63, 87]
weights = [.20, .20, .20, .40]  # can also use [1.0, 1.0, 1.0, 2.0]
                                # since it has the same proportions
weighted_mean = sum(s * w for s, w in zip(sample, weights)) / sum(weights)
# print(weighted_mean) # prints 81.4

"""3-4. calculating the median in python"""
from typing import List
sample = [0, 1, 5, 7, 9, 10, 14]

def get_median(
  values: List[float]
) ->      float:
  length = len(values)
  mid_index = int(length / 2) # int will floor decimals
                              # also, remember that python indexes start at 0

  ordered_values = sorted(values)

  if length % 2 == 0:
    return (ordered_values[mid_index - 1] + ordered_values[mid_index]) / 2.0
  else:
    return ordered_values[mid_index]

# print(get_median(sample)) # prints 7  

"""3-5. calculating the mode in python"""
from collections import defaultdict

sample = [1, 3, 2, 5, 7, 0, 2, 3]

def get_mode(
  values: List[int]
) ->      int:
  number_counts = defaultdict(int) # int() produces 0

  for number in values:
    number_counts[number] += 1

  max_count = max(number_counts.values())
  modes = [ number
            for number in set(values) # sets only preserve distinct elements 
            if number_counts[number] == max_count ]

  return modes

# print(get_mode(sample)) # [2, 3]

"""3-6. calculating population variance in python"""
data = [0, 1, 5, 7, 9, 10, 14]

def get_population_variance(
  values: List[float]
) ->      float:
  mean = sum(values) / len(values)
  population_variance = sum((v - mean)**2 for v in values) / len(values)
  return population_variance

# print(get_population_variance(data)) # prints 21.387755102040813

"""3-7. calculating population standard deviation"""
from math import sqrt

def get_population_std_dev(
  values: List[float]
) ->      float:
  return sqrt(population_variance(values))

# print(get_population_std_dev(data)) # prints 4.624689730353898

"""3-8. calculating standard deviation for a sample"""
def get_variance(
  values:     List[float],
  is_sample:  bool = False
) ->          float:
  mean = sum(values) / len(values)
  variance = sum((v - mean)**2 for v in values) / (len(values) - (1 if is_sample else 0))
  return variance

def get_std_dev(
  values:     List[float],
  is_sample:  bool = False
) ->          float:
  return sqrt(get_variance(values, is_sample))

# print(f"VARIANCE = {get_variance(data, is_sample = True)}") # 24.95238095238095
# print(f"STD DEV = {get_std_dev(data, is_sample = True)}") # 4.99523582550223

"""3-9. the normal distribution function in python"""
# normal distribution, which returns likelihood (not probability)
def get_normal_pdf(
  x:        float,
  mean:     float,
  std_dev:  float
) ->        float:
  return (1.0 / (2.0 * math.pi * std_dev**2)**0.5) * math.exp(-1.0 * ((x - mean)**2 / (2.0 * std_dev**2)))

"""3-10. the normal distribution CDF in python"""
from scipy.stats import norm

mean = 64.43
std_dev = 2.99

area_up_to_mean = norm.cdf(64.43, mean, std_dev)
# print(area_up_to_mean) # prints 0.5

"""3-11. getting a middle range probability using the CDF"""
middle_area = norm.cdf(66, mean, std_dev) - norm.cdf(62, mean, std_dev)

# print(middle_area) # prints 0.4920450147062894

"""3-12. using the inverse CDF (called ppf()) in python"""
x_95_percent_cdf = norm.ppf(.95, loc = mean, scale = std_dev)

# print(x_95_percent_cdf) # 69.3481123445849

"""3-13. generating random numbers from a normal distribution"""
import random

for i in range(0, 1000):
  random_probability = random.uniform(0.0, 1.0)
  random_x = norm.ppf(random_probability, loc = mean, scale = std_dev)
  # print(random_x)

"""3-14. turn z-scores into x-values and vice versa"""
def get_z_score(x, mean, std_dev):
  return (x - mean) / std_dev

def convert_z_to_x(z, mean, std_dev):
  return (std_dev * z) + mean

mean = 140000
std_dev = 3000
x = 150000

z = get_z_score(x, mean, std_dev)
back_to_x = convert_z_to_x(z, mean, std_dev)

# print(f"Z-Score: {z}") # Z-Score: 3.3333333333333335
# print(f"Back to X: {back_to_x}") # Back to X: 150000.0

"""3-15. exploring the central limit theorem"""
# samples of the uniform distribution will average out to a normal distribution
import plotly.express as px

sample_size = 31 # normal distributions require 31 or more
sample_count = 1_000

# central limit theorem, 1_000 samples each with 31 random numbers between 0.0 and 1.0
x_values = [sum(random.uniform(0.0, 1.0) 
                for i in range(sample_size)) 
              / sample_size
            for _ in range(sample_count)]

y_values = [1 for _ in range(sample_count)]

# px.histogram(x = x_values, y = y_values, nbins = 20).show()

"""3-16. retrieving a critical z-value for a normal distribution"""
def get_lower_upper_tail_areas(
  probability:  float
) ->            (float, float):
  lower_tail_area = (1.0 - probability) / 2.0
  upper_tail_area = 1.0 - lower_tail_area

  return (lower_tail_area, upper_tail_area)

def get_normal_dist_critical_z_values(
  probability:  float
) ->            (float, float):
  """
  https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.stats.norm.html
  """
  norm_dist = norm( 
                loc = 0.0,  # specifies the mean (default 0)
                scale = 1.0 # specifies the std dev (default 1)
              )
  lower_tail_area, upper_tail_area = get_lower_upper_tail_areas(probability)

  return norm_dist.ppf(lower_tail_area), norm_dist.ppf(upper_tail_area)

# print(get_normal_dist_critical_z_values(probability = .95))
# (-1.959963984540054, 1.959963984540054)

"""3-17. calculating a confidence interval in python"""
def get_confidence_interval(
  probability:  float,
  sample_mean:  float,
  sample_std:   float,
  sample_size:  int     # must be greater than 30
) ->            (float, float):
  lower_z, upper_z = get_normal_dist_critical_z_values(probability)
  lower_ci = lower_z * (sample_std / sqrt(sample_size))
  upper_ci = upper_z * (sample_std / sqrt(sample_size))
  """
  interval is inversely proportional to sample size
  larger samples increase our confidence in a smaller range
  """

  return sample_mean + lower_ci, sample_mean + upper_ci

# print(
#   get_confidence_interval(
#         probability = .95,
#         sample_mean = 64.408,
#         sample_std = 2.05,
#         sample_size = 31
#   )
# ) # (63.68635915701992, 65.12964084298008)

"""3-18. calculating the probability of recovery in 15-21 days"""
# cold has 18 day mean recovery, 1.5 std dev
min_x = 15
max_x = 21
mean = 18
std_dev = 1.5

probability =   norm.cdf(max_x, mean, std_dev) \
              - norm.cdf(min_x, mean, std_dev)

# print(probability) # 0.9544997361036416

"""3-19. python code for getting x-value with 5% of area behind it"""
# what x-value has 5% of area behind it?
x_5_percent_cdf = norm.ppf(.05, mean, std_dev)

# print(x_5_percent_cdf) # 15.53271955957279

"""3-20. calculating the one-tailed p-value"""
# probability of 16 or less days
p_value = norm.cdf(16, mean, std_dev)

# print(p_value) # 0.09121121972586788

"""3-21. calculating a range for a statistical significance of 5%"""
# what x-value has 2.5% of area behind it?
x_2_pt_5_percent_cdf = norm.ppf(.025, mean, std_dev)

# what x-value has 97.5% of area behind it?
x_97_pt_5_percent_cdf = norm.ppf(.975, mean, std_dev)

# print(x_2_pt_5_percent_cdf) # 15.060054023189918
# print(x_97_pt_5_percent_cdf) # 20.93994597681008

"""3-22. calculating the two-tailed p-value"""
# probability of 16 or less days
probability_up_to_16 = norm.cdf(16, mean, std_dev)

# probability of 20 or more days"""
# since the mean is 18, 
# 20 is the symmetrical equivalent value on the other end
probability_20_or_more = 1.0 - norm.cdf(20, mean, std_dev)

# p-value of both tails
p_value = probability_up_to_16 + probability_20_or_more

# print(p_value) # 0.18242243945173575

"""3-23. getting a critical value range with a t-distribution"""
from scipy.stats import t

# get critical value range for 95% confidence
# with a sample size of 25
# (t-distributions work with sample sizes of 30 or less)
sample_size = 25
probability = .95
degrees_of_freedom = sample_size - 1
lower_tail_area, upper_tail_area = get_lower_upper_tail_areas(probability)

"""
https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.stats.t.html
"""
lower_critical_z = t.ppf(lower_tail_area, df = degrees_of_freedom)
upper_critical_z = t.ppf(upper_tail_area, df = degrees_of_freedom)

# print(lower_critical_z, upper_critical_z)
# -2.0638985616280205 2.0638985616280205

"""Exercises"""
# 1. You bought a spool of 1.75 mm filament for your 3D printer. You want to measure how close the filament diameter really is to 1.75 mm. You use a caliper tool and sample the diameter five times on the spool:
# 1.78, 1.75, 1.72, 1.74, 1.77
# Calculate the mean and standard deviation for this set of values.
# def std_dev(
#   values:     List[float],
#   is_sample:  bool = False
# ) ->          float:
#   return sqrt(variance(values, is_sample))

samples = [1.78, 1.75, 1.72, 1.74, 1.77]
mean = sum(samples) / len(samples)
variance = get_variance(samples, is_sample = False)
std_dev = get_std_dev(samples, is_sample = False)

# print(f"MEAN = {mean}") # 1.752
# print(f"VARIANCE = {variance}") # 0.00045600000000000084
# print(f"STD DEV = {std_dev}") # 0.02135415650406264

# 2. A manufacturer says the Z-Phone smart phone has a mean consumer life of 42 months with a standard deviation of 8 months. Assuming a normal distribution, what is the probability a given random Z-Phone will last between 20 and 30 months?
mean = 42
std_dev = 8
middle_area = norm.cdf(30, mean, std_dev) - norm.cdf(20, mean, std_dev)

# print(middle_area) # 0.06382743803380352

# 3. I am skeptical that my 3D printer filament is not 1.75 mm in average diameter as advertised. I sampled 34 measurements with my tool. The sample mean is 1.715588 and the sample standard deviation is 0.029252.
# What is the 99% confidence interval for the mean of my entire spool of filament?
# print(
#   get_confidence_interval(
#     probability = .99,
#     sample_size = 34,
#     sample_mean = 1.715588,
#     sample_std = 0.029252
#   )
# )
# (1.7026658973748656, 1.7285101026251342)

# 4. Your marketing department has started a new advertising campaign and wants to know if it affected sales, which in the past averaged $10,345 a day with a standard deviation of $552. The new advertising campaign ran for 45 days and averaged $11,641 in sales.
# Did the campaign affect sales? Why or why not? (Use a two-tailed test for more reliable significance.)
mean = 10_345
std_dev = 552

# probability of $9_049 in sales or less
# since the mean is 10_345, 
# 9_049 is the symmetrical equivalent value on the other end
probability_up_to_9_049 = norm.cdf(9_049, mean, std_dev)

# probability of $11_641 in sales or more"""
probability_11_641_or_more = 1.0 - norm.cdf(11_641, mean, std_dev)

# p-value of both tails
# p_value = probability_up_to_9_049 + probability_11_641_or_more
p_value = probability_11_641_or_more * 2 # take advantage of symmetry

# print(p_value) # 0.01888333596496139
