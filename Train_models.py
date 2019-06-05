# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 20:45:20 2019

@author: akovacevic
"""
####### Neophodni paketi 

"""
!pip install imblearn
"""
### Set working directory
import os
#os.chdir("F:/Kursevi i Ucenje/DS Task")

############# Imput parameters
path = 'F:/Kursevi i Ucenje/DS Task'
filename = 'campaign_821471.log'
fullpath = 'F:/Kursevi i Ucenje/DS Task' + '/' + filename 
test_size = 0.2 # !!!!!!!!!!!!!!!!!!!!!!!!! da li da bude parametar
models = ['LogisticRegression', 'RandomForest', 'XGBoost']
#model = 'LogisticRegression'

df_dict = {'df1':'campaign_821471.log',
           'df2':'campaign_768874.log',
           'df3':'creative_5188417.log'}
################# Program logic

###### imput requred packages and libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as pl
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.under_sampling import ClusterCentroids
import pickle

from Basic_Preparation import MissingImputer
from Basic_Preparation import FunctionBinning
from Basic_Preparation import RemoveFeatures
from Feature_Engineering import Feature_Quantification
import my_scorer
from sklearn.metrics import precision_recall_curve, make_scorer, auc



##### create param_grid dictionaries for weighted classes
param_grid_lr_wc = {
    'classify__penalty':['l1'],
    'classify__class_weight': [{0:0.02, 1:0.98} ,{0:0.1, 1:0.9}],
    'classify__C' : [0.03, 0.05, 0.1] }    

param_grid_rf_wc = {
    'classify__n_estimators':[50, 150, 300],
    'classify__max_depth':[2, 4, 6],
    'classify__class_weight': ['balanced']#[{0:0.02, 1:0.98}, {0:0.1, 1:0.9}] 
    }    

param_grid_gbm_wc = {
    'classify__n_estimators': [ 50, 150, 300],
    'classify__boostig_type': ['gbdt'],
    'classify__objectiv': ['binary'],
    'classify__max_depth' : [2, 4, 6],
    'classify__class_weight': ['balanced'],
    'classify__num_leaves':[15, 30, 50],
    'classify__learning_rate':[0.03, 0.1, 0.3]}    
   
##### create param_grid dictionaries for Imbalased classes
param_grid_lr_im = {
    'classify__penalty':['l1'],
    'classify__C' : [0.03, 0.05, 0.1, 0.7] }    

param_grid_rf_im = {
    'classify__n_estimators':[50, 150, 300],
    'classify__max_depth':[2, 4, 6]}
       

param_grid_gbm_im = {
        'classify__n_estimators': [ 50, 150, 300],
    'classify__boostig_type': ['gbdt'],
    'classify__objectiv': ['binary'],
    'classify__max_depth' : [2, 4, 6],
    'classify__class_weight': ['balanced'],
    'classify__num_leaves':[15, 30, 50],
    'classify__learning_rate':[0.03, 0.1, 0.3]    }

numerical = ['hour', 'day_of_week', 'install_week', 'ad_blocker']
categorical = ['advertiser_id', 'platform', 'network', 'request_tld', 'creative_id', 'state', 'project_id', 'campaign_id', 'dma', 
               'location_id', 'keyword_id', 'url_category_ids', 'organization', 'browser_ver', 'country_code', 'content_category_ids']
convert_dict = {}
for att in categorical:
    convert_dict.update ({att: object})
    
    ####### crate pipe with preprocesing 
pipeLR_wc = Pipeline( [('missing', MissingImputer(cat_features = categorical, num_features = numerical) ) ,
                 ('removeFeatures', RemoveFeatures(onlyStr = False)),
                 ('binning', FunctionBinning( treshold = 1) ),
                 ('quantifyFeatures', Feature_Quantification()), #])
                 ('classify', LogisticRegression(solver = 'liblinear', max_iter = 300 ) )])
    
pipeRF_wc = Pipeline( [('missing', MissingImputer(cat_features = categorical, num_features = numerical) ) ,
                 ('removeFeatures', RemoveFeatures(onlyStr = False)),
                 ('binning', FunctionBinning( treshold = 1) ),
                 ('quantifyFeatures', Feature_Quantification()), #])
                 ('classify', RandomForestClassifier() )])
    
pipeGBM_wc = Pipeline( [('missing', MissingImputer(cat_features = categorical, num_features = numerical) ) ,
               ('removeFeatures', RemoveFeatures(onlyStr = False)),
               ('binning', FunctionBinning( treshold = 1) ),
               ('quantifyFeatures', Feature_Quantification()) , #])
               ('classify', LGBMClassifier(n_jobs = 4) )])    
    
    
pipeLR_im = pl( [('missing', MissingImputer(cat_features = categorical, num_features = numerical) ) ,
                 ('removeFeatures', RemoveFeatures(onlyStr = False)),
                 ('binning', FunctionBinning( treshold = 1) ),
                 ('quantifyFeatures', Feature_Quantification()), #])
                 ('underSample', ClusterCentroids(sampling_strategy = 1.0)),
                 ('classify', LogisticRegression(solver = 'liblinear', max_iter = 300 ) )])
    
pipeRF_im = pl( [('missing', MissingImputer(cat_features = categorical, num_features = numerical) ) ,
                 ('removeFeatures', RemoveFeatures(onlyStr = False)),
                 ('binning', FunctionBinning( treshold = 1) ),
                 ('quantifyFeatures', Feature_Quantification()), #])
                 ('underSample', ClusterCentroids(sampling_strategy = 1.0)),
                 ('classify', RandomForestClassifier( ))])
    
pipeGBM_im = pl( [('missing', MissingImputer(cat_features = categorical, num_features = numerical) ) ,
               ('removeFeatures', RemoveFeatures(onlyStr = False)),
               ('binning', FunctionBinning( treshold = 1) ),
               ('quantifyFeatures', Feature_Quantification()) , #])
               ('underSample', ClusterCentroids(sampling_strategy = 1.0)),
               ('classify', LGBMClassifier(n_jobs = 4) )])    
    
    # Dictionary of pipelines and classifier types for ease of reference
    
gs_LR_wc = GridSearchCV(pipeLR_wc, param_grid = param_grid_lr_wc, 
                            n_jobs = 1,cv=3, scoring = my_scorer.prc_score, return_train_score=False)
gs_RF_wc = GridSearchCV(pipeRF_wc, param_grid = param_grid_rf_wc, 
                            n_jobs = 1,cv=3, scoring = my_scorer.prc_score, return_train_score=False)
gs_GBM_wc = GridSearchCV(pipeGBM_wc, param_grid = param_grid_gbm_wc, 
                            n_jobs = 1,cv=3, scoring = my_scorer.prc_score, return_train_score=False)
gs_LR_im = GridSearchCV(pipeLR_wc, param_grid = param_grid_lr_im, 
                            n_jobs = 1,cv=3, scoring = my_scorer.prc_score, return_train_score=False)
gs_RF_im = GridSearchCV(pipeRF_wc, param_grid = param_grid_rf_im, 
                            n_jobs = 1,cv=3, scoring = my_scorer.prc_score, return_train_score=False)
gs_GBM_im = GridSearchCV(pipeGBM_im, param_grid = param_grid_gbm_im, 
                            n_jobs = 1,cv=3, scoring = my_scorer.prc_score, return_train_score=False)


####### read data

for df in df_dict.values():
    
    data = pd.read_csv(df, sep="\t",  dtype = convert_dict)
    
    print("Whole Data Set is of shape: " + str(data.shape))
    print("Whole Data Imbalance is: " + str(data['label'].sum()/data.shape[0]))

    ####### sample data to test the flow
   # data['Weights'] = np.where(data['label'] > 0, .98, .02)
   # data = data.sample(frac=.05, weights='Weights')
    
###### split target attribut 
    X = data.loc[:, data.columns != 'label']
    y = data['label']
 
    
    ###### train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, stratify = y )
    

    """
    model_LR_wc = gs_LR_wc.fit(X_train, y_train)
    model_RF_wc = gs_RF_wc.fit(X_train, y_train)
    
    pickle._dump(model_LR_wc, open('Models/model_LR_wc.sav', 'wb'))
    pickle._dump(gs_LR_wc.cv_results_, open("cv_results_LR.pkl","wb"))
    pickle._dump(model_RF_wc, open('Models/model_RF_wc.sav', 'wb'))
    """
    
    df_name = df.split('.')[0]
    print('Data Set: %s' % df_name)
    
    grids = [gs_GBM_wc ,gs_RF_wc,  gs_LR_wc, gs_GBM_im, gs_RF_im, gs_LR_im ]
    grid_dict = {0:'GBM', 1: 'RandomForest', 2 :'LogisticRegression', 3: 'GBMWithImb', 4:'RandomForestWithImb', 5: 'LogisticWithImb'}
    
 
    
    best_auc_prc = 0.0
    best_clf = 0
    best_gs = ''
    
    for idx, gs in enumerate(grids):
        print('\nEstimator: %s' % grid_dict[idx])		
        print(gs)
        model = gs.fit(X_train, y_train)
        print('Best params: %s' % model.best_params_)
        print('Best training au_prc: %.3f' % model.best_score_)
        #model.best_estimator_
        y_pred = model.predict(X_test)
        y_probas = model.predict_proba(X_test)
        y_probas = y_probas[:,0]
        print('Test set au_prc score for best params: %.3f ' % my_scorer.au_prc(y_test, y_probas))
    	   # Track best (highest test accuracy) model
        mn = grid_dict[idx]
        model_name = mn + '_' + df_name + '.sav'
        model_filename = 'Models/' + model_name
        
        pickle._dump(model.best_estimator_, open(model_filename, 'wb'))
        dict_name = 'cv_results_' + mn + '_' + df_name + '.pkl' 
        pickle._dump(gs.cv_results_, open(dict_name, "wb"))
        
        #if prc_score(y_test, y_pred) > best_auc_prc:
         #   best_auc_prc = au_prc(y_test, y_pred)
          #  best_gs = gs
           # best_clf = idx
    print('\nClassifier with best test set metrics: %s' % grid_dict[best_clf])

"""    
y_test
p = model.predict(X_test)
pp = model.predict_proba(X_test)
pp = pp[:,0]

au_prc(y_test, pp)
model
"""
####  try LGBMClassifier without preprocessing
"""
options = {
        'n_estimators': 10,
        'learning_rate': 1.0,
        'objective': 'binary',
        'min_data': 1,
        'min_data_in_bin': 1,
        'random_state': 0,
        'verbose': -1,
}
m = LGBMClassifier(categorical_feature = categorical,**options)
m.fit(X_train, y_train)
"""