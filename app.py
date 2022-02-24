 
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve,classification_report,roc_curve,auc
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score,roc_auc_score

def udf_roc_auc_curve_diagram(actual,predicted):
    #print('Confusion Metrix:')
    #print(confusion_matrix(actual,predicted))

    #print('Accuracy Score: {0:.2f} ,Precision: {1:.2f}, Recall: {2:.2f} , F1 Score: {3:.2f} '.format(accuracy_score(actual,predicted),precision_score(actual,predicted),recall_score(actual,predicted),f1_score(actual,predicted)))
    st.write('Accuracy Score:', accuracy_score(actual,predicted))
    fpr, recall, thresholds = roc_curve(actual,predicted)
    roc_auc = auc(fpr, recall)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(fpr, recall)
    plt.plot([0, 1], [0, 1], 'o--')
    plt.title('AUC = %0.2f' % roc_auc)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    st.pyplot(plt)
    
upload_file=st.file_uploader("Select a file:")

if upload_file is not None:
    
    df=pd.read_csv(upload_file)
    #st.write(df)

# loading the trained model
    #pickle_in = open('classifier.pkl', 'rb') 
    classifier = load('model.pkl')
    
    X_test=df[df.columns[1:202]]
    y_test=df['target']
    
    y_pred=classifier.predict(X_test)
    
    
    udf_roc_auc_curve_diagram(y_test,y_pred)
    st.write(X_test.head())
    st.write(y_test.head())
    
  
