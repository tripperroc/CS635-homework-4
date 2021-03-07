import math
import pandas as pd
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt

from descender import descend


def h(x, p):
	""" return a linear function of x, with
	x -- an array in the data space, where len(x) = len(p) - 1
	p -- the parameters of linear function, where p[0] is the coefficent of x[0] etc. and p[-1] is the bias parameter.
	"""
	return np.append(x,[1]).T @ p @ np.append(x,[1])

""" extract train and test data. This should be identical to the code used in linear. If you like,
you may factor this into a function in descender.py that both programs can use.
"""

def g(p, x, y):
	"""[5 points] return the gradient of h for parameter list p
	p -- the parameters of linear function, where p[0] is the coefficent of x[0] etc. and p[-1] is the bias parameter.
	x_batch -- a matrix representing a batch of input the data space, where len(x) = len(p) - 1
	y_batch -- the (real number) label associated with the items in the data batch
	"""

def g_regularized(p, x_batch, y_batch, l):
	"""[5 points] return the gradient of h for parameter list p
	p -- the parameters of linear function, where p[0] is the coefficent of x[0] etc. and p[-1] is the bias parameter.
	x_batch -- a matrix representing a batch of input the data space, where len(x) = len(p) - 1
	y_batch -- the (real number) label associated with the items in the data batch
	"""

"""[5 points] randomly initialize your parameters. You may want to store them as an upper triangular
   matrix rather than a vector, because doing so can make computing g and h easier, and if descend is
   written correct it should be able to handle it.
"""


"""[5 points] Run descent for 0, 5, 100, 200, and 1600 iterations on the training data only. 
[5 points] On both the test and training data, graph sse against each of these running times
"""

	








