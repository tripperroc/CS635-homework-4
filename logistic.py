import math
import pandas as pd
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt

from descender import descend


def h(x, p):
	"""[5 points] return the logistic function on x, with
	x -- an array in the data space, where len(x) = len(p) - 1
	p -- the parameters of logistic function, where p[0] is the coefficent of x[0] etc. and p[-1] is the bias parameter.
	"""
	

def g(p, x_batch, y_batch):
	"""[5 points] return the gradient of h for parameter list p
	p -- the parameters of linear function, where p[0] is the coefficent of x[0] etc. and p[-1] is the bias parameter.
	x_batch -- a matrix representing a batch of input the data space, where len(x) = len(p) - 1
	y_batch -- the (real number) label associated with the items in the data batch
	"""
	

"""extract data and labels from mnist data, and convert labels to binary, where 0 = 1 and every other label is a 1.
So effectively the goal is solely to learn to distinguish 0 from other digits. 
You may need to change the directory path to where you store your data
"""
yx = np.array(pd.read_csv("../mnist/train.csv"))
y = yx[:,0]
y = [0 if yi==0 else 1 for yi in y]

x = yx[:,1:]


"""[5 points] Initialize parameters. This should be done randomly, and you should experiment with different reasonable approaches
	For debugging purposes you may want to fix a constant seed for the random number generator. You may want to store the 
	parameters as an upper triangular matrix, because if you do it properly it easy to compute h and g.
"""

"""[5 point] Run descent for 0, 1, 5, 10, 25, 50, and 400 iterations. 
[5 points] Graph precision, recall, and f1 for correctly guessing the zero class on your training data by
rounding the output of h(x,p) to 0 or 1, whichever is closest. You can put all three quantities in one graph
or create three separate graphs. 
"""








