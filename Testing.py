# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 21:39:54 2019

@author: akovacevic
"""
import os
import numpy as np
import pandas as pd
import pickle 
import my_scorer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, precision_recall_curve, auc, confusion_matrix, classification_report
from mlxtend.plotting import  plot_confusion_matrix
import seaborn as sns

def confusion_matrix_plot(y_true, y_pred, labels):
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    annot = np.array(cm)
    nrows, ncols = cm.shape
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    plt.show()
    
#### Imput parameters
df = input("For wich data you want to test results for conversion rate? (Possible choices: ['campaign_821471', 'campaign_768874','creative_5188417' ] )! ")
#df = 'campaign_821471'
metric = 'au_prc'
df_list = ['campaign_821471', 'campaign_768874','creative_5188417']

if df not in df_list:
    print('Your imput is not correct, please try again')
    print(df)

else:

    #### Create all variables 
    df_file = df + '.log'
    numerical = ['hour', 'day_of_week', 'install_week', 'ad_blocker']
    categorical = ['advertiser_id', 'platform', 'network', 'request_tld', 'creative_id', 'state', 'project_id', 'campaign_id', 'dma', 
                   'location_id', 'keyword_id', 'url_category_ids', 'organization', 'browser_ver', 'country_code', 'content_category_ids']
    convert_dict = {}
    for att in categorical:
        convert_dict.update ({att: object})
    
    data = pd.read_csv(df_file, sep="\t",  dtype = convert_dict)
    
    X = data.loc[:, data.columns != 'label']
    y = data['label']
         
        ###### train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y )
    
    
    # read all models for the given data set
    end = df.split('_')[1]    
    items = os.listdir("Models/.") #########!!!!!!!!!!!!!!!!!!! promeni putanju
    models = []
    for names in items:
        if names.endswith(end + '.sav'):
            models.append(names)
    #print (models)
    
    result = pd.DataFrame(columns = ['DataSet', 'Model', 'score'])
    probas_dict = {}
    pred_dict = {}
    for mod in models:
        filename = mod
        #filename = 'lraaa' #'modeli/LogisticRegression_campaign_821471.sav'
        loaded_model = pickle.load(open( 'Models/' + mod, 'rb')) #!!!!!!!!!!!!!!!!!!!!!!!!!!! PUTANJU VRATI   
        pred = loaded_model.predict(X_test)
        pred_proba = loaded_model.predict_proba(X_test)
        pred_proba = pred_proba[:, 1]
        score = my_scorer.au_prc(y_test, pred_proba)
        #precision, recall, thresholds = precision_recall_curve(y_test, pred_proba)
        result = result.append({'DataSet': df_file, 'Model': mod, 'score': score},  ignore_index=True) 
        probas_dict.update({mod : pred_proba})
        pred_dict.update({mod : pred})
    
    #max(d, key=d.get)
    ####### select best model results
    best_score = result['score'].max()
    best_model = result.loc[result['score'] == best_score ,'Model'].iloc[0]
    print('Best model for selected data set is: ' + best_model + '. Score au_prc is: ' + str(best_score))
    print('-------------------------------------------------' )
    pred_proba = probas_dict[best_model]
    pred = pred_dict[best_model]
    
    #plot confusion matrix
    #confusion_matrix_plot(y_test, pred, labels = loaded_model.classes_)
    
    precision, recall, thresholds = precision_recall_curve(y_test, pred_proba)
    
    ######## Plot Precision-Recall curve 
    plt.close()
    plt.step(recall, precision, color='b', alpha=0.2,  where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve') #AP={0:0.2f}'.format(average_precision))
    
    
    confusion_matrix_plot(y_test, pred, labels =np.array([0,1]))
    print(classification_report(y_test, pred))
    """
    precision, recall, _ = precision_recall_curve(y_test, pred_proba)
    
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    from inspect import signature
    from sklearn.metrics import average_precision_score
    average_precision = average_precision_score(y_test, pred_proba)

    print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    """
    """
    cm = confusion_matrix(y_test, pred, labels= np.array([0,1]))
    annot = np.array(cm)
    nrows, ncols = cm.shape
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    plt.show()
    plt.close()
"""

"""
import sklearn
sklearn.__version__
!pip install scikit-learn==0.20.0
loaded_model = pickle.load(open( 'Models/RandomForest_campaign_821471.sav', 'rb'))
pred_proba = loaded_model.predict_proba(X_test)
pred_proba = pred_proba[:, 1]
pred = loaded_model.predict(X_test)
plot_confusion_matrix(confusion_matrix(y, pred), cmap=plt.cm.Greys)

!pip install lightgbm
"""


