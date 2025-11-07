import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

df = pd.read_csv('creditcard.csv')

X = df.drop(['Class', 'Time'], axis=1, errors='ignore')
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
clf.fit(X_train_res, y_train_res)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]

print(classification_report(y_test, y_pred))
print('AUC:', roc_auc_score(y_test, y_proba))
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))

joblib.dump(clf, 'rf_fraud_model.joblib')
print('Model saved to rf_fraud_model.joblib')
