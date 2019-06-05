# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 11:27:42 2019

@author: akovacevic
"""
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator, clone


class Feature_Quantification(BaseEstimator, TransformerMixin):
    '''
    This one is for mean - it can be used for binary or numerical class
    In case of binary it is interpreted as ratio of positive class
    '''
    REQUIREMENTS = [
        'pandas',  # install specific version of a package
        'scikit-learn',  # install latest version of a package
        'numpy'  # install latest version of a package
    ]

    def __init__(self):
        
        self.featurizers = {}
        

    def fit(self, X, y=None):
        '''
        Creates map between values of categorical features and corresponding quantifications
        :param X:
        :param y:
        :return:
        '''
        
        m = X.shape[0]
        expected_value = y.sum()/m
        
        categorical_features = list(X.select_dtypes(include='object').columns)
        for att in categorical_features:
            quantified = pd.concat([X[att], y], axis=1)
            grouped_by_label = quantified.groupby(att).agg('mean') # this one is aggregation it can be change with other methods
            value_dist = X[att].value_counts()/m
            value_dist_df = pd.DataFrame({att:value_dist.index, 'proc':value_dist.values})
            all_df_att = value_dist_df.set_index(att).join(grouped_by_label, how='inner')
            all_df_att['combine'] = all_df_att.proc * all_df_att.label
            self.featurizers.update({att : all_df_att})
        self.featurizers.update({'expected_value' : expected_value})
        
        #print('Feature_Quantification Fitting is done.')
        return self

    def transform(self, X, y = None):
        #Do transformations
        
        expected_value = self.featurizers['expected_value']
        
        for key, val in self.featurizers.items():
            if key == 'expected_value':
                continue
            else:
                map_df = val
                X = pd.merge(X, map_df, how = 'left', left_on = key, right_index=True)
                X.loc[:, ['proc', 'label', 'combine']].fillna(expected_value, inplace=True)
                X.rename(columns={'proc': key + '_proc', 'label': key + '_label', 'combine' : key + '_combine' }, inplace=True)
           
                X = X.drop(key, axis=1)
      
        #Sprint('Feature_Quantification transformation is done.')
        return X
