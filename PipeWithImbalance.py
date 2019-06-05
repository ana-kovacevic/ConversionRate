# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 17:19:28 2019

@author: akovacevic
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as pl
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import pickle


def au_prc(y_true, y_pred):
    prec, recall, tresholds = precision_recall_curve(y_true, y_pred)
    prc_score = auc( recall, prec)

    return prc_score

prc_score = make_scorer(au_prc, greater_is_better=True)

#### read data
campagin1 = pd.read_csv('campaign_821471.log', sep="\t")

##### separate target value 
X = campagin1.iloc[:,1:]
y = campagin1.iloc[:,0]

non_categorical = set(['hour', 'day_of_week', 'install_week', 'ad_blocker'])
categorical = set(X.columns)
categorical = list(categorical - non_categorical)
numerical = list(non_categorical)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y )

from Basic_Preparation import MissingImputer
from Basic_Preparation import FunctionBinning
from Basic_Preparation import SetCategoricalFeatureTypes
from Basic_Preparation import RemoveFeatures
from sklearn.ensemble import GradientBoostingClassifier
from Feature_Engineering import Feature_Quantification
from sklearn.metrics import make_scorer, precision_recall_curve, auc, confusion_matrix, classification_report
from mlxtend.plotting import  plot_confusion_matrix
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids


pipeGBC = pl( [('missing', MissingImputer(cat_features = categorical, num_features = numerical) ) ,
                         ('setTypes', SetCategoricalFeatureTypes(cat_features = categorical) ),
                         ('binning', FunctionBinning(cat_features= categorical, treshold = 1) ),
                         ('removeFeatures', RemoveFeatures(onlyStr = False)),
                         ('quantifyFeatures', Feature_Quantification()), #])
                         ('underSample', ClusterCentroids()),
                         ('classify', GradientBoostingClassifier() )])
    
param_grid_gbc = {
    'underSample__sampling_strategy' : [1.0, 1.5, 2.5],
    'classify__n_estimators':[50,100],
    'classify__max_depth':[3,5] }    

gs_GBC = GridSearchCV(pipeGBC, param_grid = param_grid_gbc, n_jobs = 1,cv=4, scoring = prc_score, return_train_score=False)
model_gbc = gs_GBC.fit(X_train, y_train)

pickle._dump(model_gbc, open('Models/First_GBC_IM.sav', 'wb'))

#LR_IM = pipeGBC.fit(X_train, y_train)
pred = model_gbc.predict(X_train)
pred_proba = model_gbc.predict_proba(X_train)

probs = pred_proba[:, 1]

# calculate roc curve
precision, recall, thresholds = precision_recall_curve(y_train, probs)

######## Plot Precision-Recall curve 
plt.step(recall, precision, color='b', alpha=0.2,  where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve') #AP={0:0.2f}'.format(average_precision))

probs.sum()/len(probs)
pred.sum()/len(pred)
model_gbc.score(X_train, y_train)
print(classification_report(y_train, pred))
plot_confusion_matrix(confusion_matrix(y_train, pred))

gs_GBC.cv_results_
