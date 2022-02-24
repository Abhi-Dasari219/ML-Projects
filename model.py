# Import All the libraries used in this project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve,classification_report,roc_curve,auc
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score,roc_auc_score
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler,BorderlineSMOTE,SMOTE, ADASYN


import joblib
import streamlit as st

warnings.filterwarnings("ignore")

def dataOverSampler(X,y):

  ros = RandomOverSampler(random_state=42)
  X_balanced, y_balanced = ros.fit_resample(X, y)
  X_df = pd.DataFrame(X_balanced,columns=X.columns)
  X_df['target'] = y_balanced

  return X_df

'''def createPicklefile(model):
    pickle_out = open("classifier.pkl", mode = "wb") 
    pickle.dump(model, pickle_out) 
    pickle_out.close()
'''
def main():

    # Read the train data set
    final_train = pd.read_csv('final_train.csv')

    #Balance the Final Data Frame
    df_balanced = dataOverSampler(final_train[final_train.columns[1:201]],final_train['target'])

    # Define X and Y from the data frame    
    X_train=df_balanced[df_balanced.columns[0:200]]
    y_train=df_balanced['target']

    # Train and Predict the values using LGBM Classifier
    model = lgb.LGBMClassifier(num_rounds=500,random_state=42,n_jobs=4,boosting_type='gbdt',objective='binary')

    # fit the LGBM model
    model.fit(X_train,y_train,verbose=0,eval_metric='logloss')

    # loading the trained model
    joblib.dump(model, 'model.pkl')

if __name__ == '__main__':
    main()
