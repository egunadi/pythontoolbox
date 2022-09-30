from typing import Tuple
from math import sqrt
from scipy.stats import norm

"""7-1. running an A/B test"""
def get_estimated_mean_and_stdev(
  num_trials:     int,  # N
  num_successes:  int   # n
) ->                    Tuple[float, float]:
  mean = num_successes / num_trials
  stdev = sqrt(mean * (1 - mean) / num_trials)

  return mean, stdev  # p, sigma

def get_a_b_test_statistic(
  num_trials_A:     int, 
  num_successes_A:  int, 
  num_trials_B:     int, 
  num_successes_B:  int
) ->    float:
  mean_A, stdev_A = get_estimated_mean_and_stdev(num_trials_A, num_successes_A)
  mean_B, stdev_B = get_estimated_mean_and_stdev(num_trials_B, num_successes_B)

  return (mean_B - mean_A) / sqrt(stdev_A ** 2 + stdev_B ** 2)

a_b_test_statistic = get_a_b_test_statistic(
                                              num_trials_A    = 1000,
                                              num_successes_A = 200, 
                                              num_trials_B    = 1000, 
                                              num_successes_B = 180
                                            )

# print(a_b_test_statistic) # -1.1403464899034472

"""
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
default has distribution mean = 0 and stdev = 1
"""
p_value = norm.cdf(a_b_test_statistic)

two_sided_p_value = p_value * 2

# print(two_sided_p_value) # 0.254141976542236
