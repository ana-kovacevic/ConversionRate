# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 13:00:40 2019

@author: akovacevic
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline



non_categorical = set(['hour', 'day_of_week', 'install_week', 'ad_blocker'])
categorical = set(X.columns)
categorical = list(categorical - non_categorical)
numerical = list(non_categorical)


from Basic_Preparation import MissingImputer
from Basic_Preparation import SetCategoricalFeatureTypes
from Basic_Preparation import RemoveFeatures
from Basic_Preparation import FunctionBinning
from Feature_Engineering import Feature_Quantification

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


X_train.isnull().sum()

mi = MissingImputer( categorical, numerical)
mi.fit(X_train, y_train)
X_train = mi.transform(X_train, y_train)

X_train.dtypes
st = SetCategoricalFeatureTypes(categorical)
st.fit(X_train, y_train)
X_train = st.transform(X_train,y_train)

rf = RemoveFeatures(onlyStr=False)
rf.fit(X_train, y_train)
X_train = rf.transform(X_train, y_train)

fb = FunctionBinning(treshold=1)
fb.fit(X_train, y_train)
X_train = fb.transform(X_train, y_train)

fq = Feature_Quantification()
fq.fit(X_train, y_train)
X_train = fq.transform(X_train, y_train)

X_train.max()

lr =  LogisticRegression(solver = 'saga', max_iter = 300 )
lr.fit(X_train,y_train)
lr.predict(X_train)
