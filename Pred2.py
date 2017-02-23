#Predictive Analysis on Titanic Data

import pandas as pd

""" Train df """

train_df = pd.read_csv("train.csv")

#Alone or with family
train_df['Alone']=train_df.SibSp+train_df.Parch
train_df['Alone'].loc[train_df['Alone'] >0] = 1

df = train_df[['Survived','Pclass','Sex','Age','Alone']].dropna()

X_train = df.iloc[:, 1:].values
y_train = df.iloc[:, 0].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X_train[:, 1] = labelencoder.fit_transform(X_train[:, 1])

""" Test Set """

test_df = pd.read_csv("test.csv")
#Alone or with family
test_df['Alone']=test_df.SibSp+test_df.Parch
test_df['Alone'].loc[test_df['Alone'] >0] = 1

df = test_df[['Pclass','Sex','Age','Alone']].dropna()

X_test = df.iloc[:, :].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X_test[:, 1] = labelencoder.fit_transform(X_test[:, 1])

# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""Naive Bayes"""

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)

#accuracy = (float(cm[0,0]) + cm[1,1])/(cm[0,0] + cm[0,1]+ cm[1,0] + cm[1,1])

print("\n\n1. " + str(round(sum(y_pred)*100.0/len(y_pred),2)) + " % Survive According to Naive Bayes")


"""Random Forest"""
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 116, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)

#accuracy = (float(cm[0,0]) + cm[1,1])/(cm[0,0] + cm[0,1]+ cm[1,0] + cm[1,1])
print("\n\n2. " + str(round(sum(y_pred)*100.0/len(y_pred),2)) + " % Survive According to Random Forest")

""" Support Vector Machines Classifier """
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0, gamma = 4)
classifier.fit(X_train, y_train)

 #Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)

#accuracy = (float(cm[0,0]) + cm[1,1])/(cm[0,0] + cm[0,1]+ cm[1,0] + cm[1,1])
print("\n\n3. " + str(round(sum(y_pred)*100.0/len(y_pred),2)) + " % Survive According to SVM")