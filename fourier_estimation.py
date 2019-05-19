"""
<module docstring>
"""

######## library imports ########

import pandas as pd
import numpy as np
import sympy
from math import pi as pi

######## class definition ########


class Fourier_Estimator:
    """
    for the fit function pass a pandas series as data
    """
    def __init__(self):
        self.pdf = None
        self.min_x = None
        self.max_x = None

    def fit(self, j, data):

        # get range variables
        self.min_x = data.min()
        self.max_x = data.max()

        # get the fourier data (data in transformed in frequencies)
        fourier_data = pd.Series(range(j)).apply(lambda i: self.cos_fun(i, data, self.min_x, self.max_x))

        # use mean to get the coeff. estimators
        fourier_coef = fourier_data.mean(axis = 1)

        # fix the intercept
        fourier_coef.iloc[0] = 1

        # sympy stuff
        x = sympy.symbols("x")

        coef = np.array(fourier_coef).reshape(1, -1)

        # array of the basis funcs
        funcs = np.array(pd.Series(range(len(fourier_coef))) \
        .apply(lambda i: self.cos_fun2(i, x, self.min_x, self.max_x))) \
        .reshape(-1, 1)

        pdf = np.dot(coef,funcs)[0][0]

        self.pdf = sympy.Max(0, pdf)

        self.norm_const()


    def get_curve(self, uniform = True, data = None, n= 500):
        if uniform:
            l = np.linspace(self.min_x, self.max_x, n)
            temp = sympy.lambdify(x, self.pdf, "numpy")
            return pd.Series(l).apply(lambda y: float(temp(y)))
        else:
            temp = sympy.lambdify(x,self.pdf,"numpy")
            return data.apply(lambda y: float(temp(y)))

    def norm_const(self, n = 500):
        if self.pdf == None:
            return None
        else:
            l = np.linspace(self.min_x, self.max_x, n)
            length = l[1] - l[0]
            temp = sympy.lambdify(x, self.pdf, "numpy")
            pdf_l = pd.Series(l).apply(lambda y: float(temp(y)))
            self.norm = (pdf_l * length).sum()

    def cos_fun(self, j, x, min_x, max_x):
        if j == 0:
            return pd.Series([1 / (max_x - min_x)] * len(x))
        else:
            return np.sqrt(2 / (max_x - min_x))*np.cos(pi*j*x / (max_x - min_x))

    def cos_fun2(self, j, x, min_x, max_x):
        if j == 0:
            return 1/(max_x - min_x)
        else:
            return sympy.sqrt(2 / (max_x - min_x)) * sympy.cos(pi*j*x / (max_x - min_x))
