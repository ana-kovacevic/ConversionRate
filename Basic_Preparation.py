# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:02:17 2019

@author: akovacevic
"""

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer


class SetCategoricalFeatureTypes(BaseEstimator, TransformerMixin):
    
    def __init__(self, cat_features):
        
        self.cat_features = cat_features
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        convert_dict = {}
        
        for att in self.cat_features:
            convert_dict.update ({att: object})
        X = X.astype(convert_dict)
        #print("Transformation data types is finished")
        return X

## ----------------------------------------------------------------------------------------------------- ##
        
class MissingImputer(BaseEstimator, TransformerMixin):
    """
    
    """
    def __init__(self, cat_features, num_features):
        
        
        self.cat_features = cat_features
        self.num_features = num_features
        self.imp_num = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.imp_cat = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value= 'Unknown')
    
    def fit(self, X, y = None):
        
        X_cat = X[self.cat_features]
        X_num = X[self.num_features]
        
        self.imp_cat.fit(X_cat)
        self.imp_num.fit(X_num)
        #print("Fitting of missing values finished")
        
        return self

        
    
    def transform(self, X, y = None):
        
        X_cat = X[self.cat_features]
        X_num = X[self.num_features]
        
        filled_cat = self.imp_cat.transform(X_cat)
        filled_num = self.imp_num.transform(X_num)
        
        filled_cat = pd.DataFrame(filled_cat, columns = self.cat_features)
        filled_num = pd.DataFrame(filled_num, columns = self.num_features)
        
        X_filled = pd.concat([filled_cat, filled_num], axis=1, sort=False)
        #print("Transform of missing values finished")
        
        return X_filled


## ----------------------------------------------------------------------------------------------------- ##
class FunctionBinning(BaseEstimator, TransformerMixin):
    '''
    Opis
    '''
    REQUIREMENTS = [
        'pandas',  # install specific version of a package
        'scikit-learn',  # install latest version of a package
        'numpy'  # install latest version of a package
    ]

    def __init__(self, treshold = 1):
        self.featurizers = {}
        self.cat_features = None
        self.treshold = treshold

    def fit(self, X, y=None):
        '''
        Creates map between values of categorical features and corresponding quantifications
        :param X:
        :param y:
        :return:
        '''
        self.cat_features = list(X.columns[X.dtypes == object])
        #features = list(X.columns)
        m = X.shape[0]
        
        for att in self.cat_features:
            value_dist = X[att].value_counts()/m * 100
            value_dist_df = pd.DataFrame({att:value_dist.index, 'proc':value_dist.values})
            big_list = value_dist_df[att][value_dist_df['proc'] > self.treshold].tolist()
            self.featurizers.update({att : big_list})
        #print("Fitting of Binning is finished")
        return self

    def transform(self, X, y = None):
        #Do transformations
      
        for feat, values in self.featurizers.items():
        
            replace = lambda x: x if x in values else 'Other'
            X[feat] = X[feat].apply(replace)   
        #print("Binning Transformation is finished")
        return X

## ----------------------------------------------------------------------------------------------------- ##
        
class OneHot(BaseEstimator, TransformerMixin):
    
    def __init__(self): #(self, cat_features):
        
        #self.cat_features = cat_features
        self.cat = None
        
    def fit(self, X, y = None):
        
        return self
    
    def transform(self, X, y = None):
        #X = pd.get_dummies(X,columns = self.cat_features)
        X = pd.get_dummies(X)
        return X
## --------------------------------------------------------------------------------------------------- ###

class RemoveFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self, onlyStr = True):
        
        self.onlyStr = onlyStr
        self.no_var_att = []
        #self.unique_att = []
        
    def fit(self, X, y = None):
        #self.no_var_att = self.num_of_occurrence_in_cat(X)
        dict_occure = self.num_of_occurrence_in_cat(X)
        
        for key in dict_occure.keys():
            if len(dict_occure[key]) == 1:
                self.no_var_att.append(key)
        
        return self
    
    def transform(self, X, y = None):
        X = X.drop(self.no_var_att, axis = 1)
        #print('Features has been removed.')
        return X
        
        
    def num_of_occurrence_in_cat(self, X):
        
        if self.onlyStr :
            logical_vec = (X.dtypes == object)
        else:
            logical_vec = [True] * X.shape[1]
    
        
        result_dict = {}   # pd.DataFrame(columns=['att_name','category','count'])
    
        for i in range(len(logical_vec)):
    
            if logical_vec[i] == True:
                att = X.columns[i]
                count_val = pd.DataFrame(pd.value_counts(X.iloc[:, i]))
                result_dict.update({att : count_val})
    
        return result_dict
    
    # we can return list of attributes also
    def remove_single_category(self, X, onlyStr = True):
        cat_counts = self.num_of_occurrence_in_cat(X)
        
        for key in cat_counts.keys():
            if len(cat_counts[key]) == 1:
                self.no_var_att.append(cat_counts[key].columns[0])
    
        X = X.drop(columns= self.no_var_att)
        return (X, self.no_var_att)
    
    def remove_unique_value_feature(self, X):
        cat_counts = self.num_of_occurrence_in_cat(X)
        
        m = X.shape[0]
        
        for key in cat_counts.keys():
            if len(cat_counts[key]) == m:
                self.unique_att.append(cat_counts[key].columns[0])
                
        X = X.drop(columns = self.unique_att)
        return (X, self.unique_att)
    
    