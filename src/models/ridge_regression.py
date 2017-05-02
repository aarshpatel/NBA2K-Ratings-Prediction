""" Experimenting with Ridge Regression on 2k dataset """

import sys
sys.path.append('../../src/')
import numpy as np
from sklearn.linear_model import Ridge
from commons.utils import *

ridge_parameters = {
    'alpha': [0.01, 0.1, 1.0, 10, 100]
}




