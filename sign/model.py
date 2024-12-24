import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, StackingClassifier

# Load the dataset
local_path = (os.path.dirname(os.path.realpath('__file__')))
file_name = ('data.csv')  # file of total data
data_path = os.path.join(local_path, file_name)
print(data_path)

df = pd.read_csv(r'' + data_path)
print(df)

# Define features and labels
units_in_data = 28  # no. of units in data
titles = ["unit-" + str(i) for i in range(units_in_data)]
X = df[titles]
y = df['letter']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2)

# Function to plot confusion matrix
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(12, 12))
    sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square=True, cmap='Blues')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title(title, size=15)
    plt.show()

# AdaBoost Classifier
clf_ada = AdaBoostClassifier()
clf_ada.fit(X_train, y_train)
y_pred_ada = clf_ada.predict(X_test)

print('AdaBoost Accuracy: ', accuracy_score(y_test, y_pred_ada))
print("AdaBoost Classification Report")
print(classification_report(y_test, y_pred_ada))
cm_ada = confusion_matrix(y_test, y_pred_ada)
print("AdaBoost Confusion Matrix")
print(cm_ada)
plot_confusion_matrix(cm_ada, 'Confusion Matrix of AdaBoost')

# Random Forest Classifier
clf_rf = RandomForestClassifier()
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)

print('Random Forest Accuracy: ', accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report")
print(classification_report(y_test, y_pred_rf))
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("Random Forest Confusion Matrix")
print(cm_rf)
plot_confusion_matrix(cm_rf, 'Confusion Matrix of Random Forest')

# Gradient Boosting Classifier
clf_gb = GradientBoostingClassifier()
clf_gb.fit(X_train, y_train)
y_pred_gb = clf_gb.predict(X_test)

print('Gradient Boosting Accuracy: ', accuracy_score(y_test, y_pred_gb))
print("Gradient Boosting Classification Report")
print(classification_report(y_test, y_pred_gb))
cm_gb = confusion_matrix(y_test, y_pred_gb)
print("Gradient Boosting Confusion Matrix")
print(cm_gb)
plot_confusion_matrix(cm_gb, 'Confusion Matrix of Gradient Boosting')

# Hybrid Stacking Classifier
base_estimators = [
    ('ada', clf_ada),
    ('rf', clf_rf),
    ('gb', clf_gb)
]

# Create the stacking classifier
stacking_clf = StackingClassifier(estimators=base_estimators, final_estimator=RandomForestClassifier())

# Fit the stacking classifier
stacking_clf.fit(X_train, y_train)
y_pred_stack = stacking_clf.predict(X_test)

# Evaluate the stacking classifier
print('Stacking Classifier Accuracy: ', accuracy_score(y_test, y_pred_stack))
print("Stacking Classifier Classification Report")
print(classification_report(y_test, y_pred_stack))
cm_stack = confusion_matrix(y_test, y_pred_stack)
print("Stacking Classifier Confusion Matrix")
print(cm_stack)
plot_confusion_matrix(cm_stack, 'Confusion Matrix of Stacking Classifier')