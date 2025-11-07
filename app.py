import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

@st.cache_resource
def load_model():
    return joblib.load('rf_fraud_model.joblib')

model = load_model()

st.title('🚨 Credit Card Fraud Detector — Random Forest')

uploaded = st.file_uploader('Upload CSV', type=['csv'])
if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head())

    X = df.drop(['Class', 'Time'], axis=1, errors='ignore')
    probs = model.predict_proba(X)[:,1]
    preds = (probs >= 0.5).astype(int)

    df['fraud_probability'] = probs
    df['predicted_fraud'] = preds

    flagged = df[df['predicted_fraud'] == 1].sort_values('fraud_probability', ascending=False)
    st.subheader('Flagged Transactions')
    st.dataframe(flagged)

    if 'Class' in df.columns:
        y_true = df['Class']
        y_pred = df['predicted_fraud']
        st.write('Precision:', precision_score(y_true, y_pred))
        st.write('Recall:', recall_score(y_true, y_pred))
        st.write('F1:', f1_score(y_true, y_pred))
        st.write('AUC:', roc_auc_score(y_true, probs))

    st.download_button('Download flagged', flagged.to_csv(index=False), 'flagged.csv')
