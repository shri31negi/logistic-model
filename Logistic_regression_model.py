
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
data = pd.read_csv(r"C:\Users\meena\Downloads\synthetic_sports_team_fan_loyalty_data.csv")
data

X = data.drop(columns=['loyalty'])
y = data['loyalty']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)


model = LogisticRegression(penalty='l1', solver='liblinear')


model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred

y_pred1 = model.predict(np.array([61,113576,17,17,7.674424,2	]).reshape(1,-1)) 

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
cv_scores


