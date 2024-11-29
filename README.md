# CU_MATH_ASSIGNMENT
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.stats import geom, binom, poisson, uniform

# Geometric distribution
def geometric(p, x):
    """
    Geometric distribution calculates the probability of the first success occurring on the x-th trial.
    Formula: P(X=x) = (1-p)^(x-1) * p
    """
    x = int(x)  # Ensure x is an integer
    if x < 1:
        return 0  # Probability is 0 for x < 1
    probability = (1 - p) ** (x - 1) * p
    return probability

# Binomial distribution
def binomial(n, p, x):
    """
    Binomial distribution calculates the probability of exactly x successes in n independent trials.
    Formula: P(X=x) = C(n, x) * p^x * (1-p)^(n-x)
    where C(n, x) = n! / (x! * (n-x)!)
    """
    n = int(n)  # Ensure n is an integer
    x = int(x)  # Ensure x is an integer
    if x < 0 or x > n:
        return 0  # Probability is 0 for invalid x
    combination = math.comb(n, x)  # C(n, x)
    probability = combination * (p ** x) * ((1 - p) ** (n - x))
    return probability

# Poisson distribution
def poisson_dist(mu, x):
    """
    Poisson distribution calculates the probability of exactly x events occurring in a fixed interval.
    Formula: P(X=x) = (mu^x * e^(-mu)) / x!
    """
    x = int(x)  # Ensure x is an integer
    if x < 0:
        return 0  # Probability is 0 for x < 0
    probability = (mu ** x) * (math.exp(-mu)) / math.factorial(x)
    return probability

# Uniform distribution
def uniform_dist(a, b, x):
    """
    Uniform distribution calculates the probability density for x within the range [a, b].
    Formula: f(x) = 1 / (b - a) for a <= x <= b, otherwise 0
    """
    if a <= x <= b:
        probability = 1 / (b - a)
    else:
        probability = 0  # Probability density is 0 outside the range [a, b]
    return probability
