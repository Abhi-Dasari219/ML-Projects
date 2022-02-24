 
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn import metrics
import seaborn as sns


from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve,classification_report,roc_curve,auc
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score,roc_auc_score

def udf_performance_metrics(actual,predicted):
    
    st.markdown(f'<h1 style="text-decoration: underline;color:Black;font-size:15px;">{"Performance Metrics"}</h1>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy Score", round(accuracy_score(actual,predicted),2))
    col2.metric("Precision Score", round(precision_score(actual,predicted),2))
    col3.metric("Recall Score", round(recall_score(actual,predicted),2))
    col4.metric("F1 Score",round(f1_score(actual,predicted),2))
    
def udf_roc_auc_curve_diagram(actual,predicted):
    fpr, recall, thresholds = roc_curve(actual,predicted)
    roc_auc = auc(fpr, recall)
    
    
    st.markdown(f'<h1 style="text-decoration: underline;color:Black;font-size:15px;">{"ROC AUC Curve"}</h1>', unsafe_allow_html=True)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(fpr, recall)
    plt.plot([0, 1], [0, 1], 'o--')
    plt.title('AUC = %0.2f' % roc_auc)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    st.pyplot(plt)

def udf_confusion_metrix(actual,predicted):
    
    cf_matrix = confusion_matrix(actual,predicted)
    
    plt.figure()
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Reds',fmt ='.0f')
    st.markdown(f'<h1 style="text-decoration: underline;color:Black;font-size:15px;">{"Confusion Matrix"}</h1>', unsafe_allow_html=True)
    ax.set_title('Confusion Matrix\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
    st.pyplot(plt)


def main():
    st.markdown(f'<div style="background-color:#FFFAFA;padding:30px"><h1 style="color:red;font-size:30px;">{"Santandar Customer Transaction Prediction"}</h1></div>', unsafe_allow_html=True)
    #st.title("Santandar Customer Transaction Prediction")
    st.markdown(f'<h1 style="color:Black;font-size:15px;">{"     Choose a File to Predict"}</h1>', unsafe_allow_html=True)
    upload_file=st.file_uploader(' ')

    if upload_file is not None:
    
        df=pd.read_csv(upload_file)
    

    # loading the trained model
        classifier = load('model.pkl')
        X_test=df[df.columns[1:202]]
        y_test=df['target']
    
        y_pred=classifier.predict(X_test)
    
    if st.checkbox("Performance Metrics"):

        udf_performance_metrics(y_test,y_pred)
                                  
    if st.checkbox("ROC AUC Curve"):

        udf_roc_auc_curve_diagram(y_test,y_pred)
        
    if st.checkbox("Confusion Matrix"):
        udf_confusion_metrix(y_test,y_pred)
        
if __name__ == '__main__':
    main()

    
  
