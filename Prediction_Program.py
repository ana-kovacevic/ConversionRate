# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 13:39:03 2019

@author: akovacevic
"""

import os
import pandas as pd
import pickle 
import sys

new_example = (input("Insert new example: "))
campaign = input('Input campagin for prediction: ')

def main( new_example, campaign = 'campaign_821471'):
    
    new_example= eval(new_example)

    #(assert type(new_example)) == dict) 
    #print(new_example)
    #print(type(new_example))
    df_new_example = pd.DataFrame(columns = new_example.keys())
    df_new_example = df_new_example.append(new_example, ignore_index=True)

#df = input("For wich data you want to test results for conversion rate? (Possible choices: ['campaign_821471', 'campaign_768874','creative_5188417' ] )! ")
#df = 'campaign_821471'

#metric = 'au_prc'
    df_list = ['campaign_821471', 'campaign_768874','creative_5188417']

    if campaign not in df_list:
        print('Your imput is not correct, please try again.')
        
    
    else:

    #### Create all variables 
        numerical = ['hour', 'day_of_week', 'install_week', 'ad_blocker']
        categorical = ['advertiser_id', 'platform', 'network', 'request_tld', 'creative_id', 'state', 'project_id', 'campaign_id', 'dma', 
                       'location_id', 'keyword_id', 'url_category_ids', 'organization', 'browser_ver', 'country_code', 'content_category_ids']
        convert_dict = {}
        for att in categorical:
            convert_dict.update ({att: object})
        for att in numerical:
            convert_dict.update ({att: int})
        
        df_new_example = df_new_example.astype(convert_dict)
        
    
        
        # read all models for the given data set
        end = campaign.split('_')[1]   
        items = os.listdir("Models/.") #########!!!!!!!!!!!!!!!!!!! promeni putanju
        models = []
        for names in items:
            if names.endswith(end + '.sav'):
                models.append(names)
        #print (models)
        
        #result = pd.DataFrame(columns = ['Model', 'pred', 'pred_proba'])
        #probas_dict = {}
        #pred_dict = {}
        prediction_dict =  {}
        for mod in models:
            
            #filename = 'lraaa' #'modeli/LogisticRegression_campaign_821471.sav'
            loaded_model = pickle.load(open( 'Models/' + mod, 'rb')) #!!!!!!!!!!!!!!!!!!!!!!!!!!! PUTANJU VRATI   
            loaded_model.predict(df_new_example)
            pred = loaded_model.predict(df_new_example)
            pred_proba = loaded_model.predict_proba(df_new_example)
            pred_proba = pred_proba[:, 1]
            #score = my_scorer.au_prc(y_test, pred_proba)
            #precision, recall, thresholds = precision_recall_curve(y_test, pred_proba)
            #result = result.append({'Model': mod, 'pred': pred, 'pred_proba': pred_proba},  ignore_index=True) 
            prediction_dict.update({mod:pred, mod + '- proba': pred_proba})
            #probas_dict.update({mod : pred_proba})
            #pred_dict.update({mod : pred})
        
        return prediction_dict

#res = predict_for_new_example(new_example)

if __name__ == '__main__':
    sys.exit(main( new_example, campaign))
    

"""
new_example = {'advertiser_id': 276, 'campaign_id': 821471, 'creative_id': 11591213, 'keyword_id': 16612975,
     'country_code': 'US', 'state':'PA', 'dma': 622, 'organization': 'Spectrum', 
       'browser_ver': 'Chrome', 'platform': 'Windows', 'network': 'W151', 'location_id': 5779, 
        'request_tld' : 'search.yahoo.com', 'ad_blocker': 0,
       'hour': 21, 'day_of_week': 5, 'install_week': 400, 'content_category_ids': 1802,
       'url_category_ids':'802|801', 'project_id': 146}
"""