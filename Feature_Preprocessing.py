# -*- coding: utf-8 -*-
"""
Created on Wed May 29 23:42:09 2019

@author: akovacevic
"""
from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.preprocessing import OneHotEncoder





        

class FeaturesForDummyEncoding(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.categories = None
        self.categorical_features = None
        
    def fit(self, X, y = None):
        self.categories = []
        g = X.columns.to_series().groupby(X.dtypes).groups
        g  = {k.name: v for k, v in g.items()}
        self.categorical_features = list(g['object'])
        
        for name in self.categorical_features:
            globals()[name] = list(X[name].unique())
            self.categories.append(globals()[name])
        
        return self.categorical_features, self.categories
    
    def transform(self, X, y = None):
                
        return self.categorical_features, self.categories
    
    
        
    
class DummyCoding_OneHotEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        
        self.enc = OneHotEncoder( handle_unknown='ignore')
        self.fit_enc = None
        
        
    def fit(self, X, y = None):
        
        X_cat = X.select_dtypes(include = ['object'])
        #X_cat = X.loc[:, X.dtypes == np.object]
        #X_other = X.select_dtypes(exclude=['object'])
        
        self.fit_enc = self.enc.fit(X_cat)
        
        return self
    
    def transform(self, X, y = None):
        
       X_cat = X.select_dtypes(include=['object'])
       X_other = X.select_dtypes(exclude=['object'])
       
       X_cat = self.fit_enc.transform(X_cat)
       X = pd.concat([X_cat, X_other], axis=1, sort=False) # ignore_index=True
       
       return X
