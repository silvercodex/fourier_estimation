import pandas as pd
import numpy as np
import sympy


class Fourier_Estimator:
    """
    for the fit function pass a pandas series as data
    """
    def __init__(self):
        self.pdf = None
    
    def fit(self,j,data):
        self.min_x = data.min()
        self.max_x = data.max()
        fourier_data = pd.Series(range(30)).apply(lambda j: self.cos_fun(j,data,self.min_x,self.max_x))

        fourier_coef = fourier_data.mean(axis = 1)
        fourier_coef.iloc[0] = 1
        x = sympy.symbols("x")
        pdf = self.pdf_fun(data,min_x,max_x)
        self.pdf = sympy.Max(0,pdf)
        self.norm_const()
        
    def get_curve(self, uniform = True, data = None, n= 500):
        if uniform:
            l = np.linspace(self.min_x,self.max_x,n)
            return pd.Series(l).apply(lambda y: float(self.pdf.subs(x,y)))
        else:
            return data.apply(lambda y: float(self.pdf.subs(x,y)))
            
    def norm_const(self, n=  500):
        if self.pdf == None:
            return None
        else:
            l = np.linspace(self.min_x,self.max_x,n)
            length = l[1]-l[0]
            pdf_l = pd.Series(l).apply(lambda y: float(self.pdf.subs(x,y)))
            self.norm = (pdf_l*length).sum()
            
    def cos_fun(self,j,x,min_x,max_x):
        if j == 0:
            return pd.Series([1/(max_x-min_x)]*len(x))
        else:
            return np.sqrt(2/(max_x-min_x))*np.cos(pi*j*x/(max_x-min_x))
    
    def cos_fun2(self,j,x,min_x,max_x):
        if j == 0:
            return 1/(max_x-min_x)
        else:
            return sympy.sqrt(2/(max_x-min_x))*sympy.cos(pi*j*x/(max_x-min_x))   
    
    def pdf_fun(self,x,min_x,max_x):
        coef = np.array(fourier_coef).reshape(1,-1)
        funcs = np.array(pd.Series(range(len(fourier_coef))).apply(lambda i: cos_fun2(i,x,min_x,max_x))).reshape(-1,1)
        return np.dot(coef,funcs)[0][0]
