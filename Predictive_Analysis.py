#Predictive Analysis on Titanic Data

import pandas as pd

titanic_df = pd.read_csv("train.csv")

#Define a function to classify 'person' as male, female, boy girl
def male_female_child(passanger):
    age,sex = passanger
    
    if age < 16:
        if sex == 'male':
            return 'boy'
        else:
            return 'girl'
    else:
        return sex
        
#Use the function
titanic_df['person'] = titanic_df[['Age','Sex']].apply(male_female_child,axis=1)

#Alone or with family
titanic_df['Alone']=titanic_df.SibSp+titanic_df.Parch
titanic_df['Alone'].loc[titanic_df['Alone'] >0] = 1

df = titanic_df[['Survived','Pclass','Sex','Age','Alone']].dropna()

X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:, 1])
#onehotencoder = OneHotEncoder(categorical_features = [2])
#X = onehotencoder.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

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
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

accuracy = (float(cm[0,0]) + cm[1,1])/(cm[0,0] + cm[0,1]+ cm[1,0] + cm[1,1])

print("\n\n1. Naive Bayes Accuracy = " + str(round(100*accuracy,2)))



"""Random Forest"""
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 116, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

accuracy = (float(cm[0,0]) + cm[1,1])/(cm[0,0] + cm[0,1]+ cm[1,0] + cm[1,1])
print("\n2. Random Forest Accuracy = " + str(round(100*accuracy,2)))


""" Support Vector Machines Classifier """
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0, gamma = 4)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

accuracy = (float(cm[0,0]) + cm[1,1])/(cm[0,0] + cm[0,1]+ cm[1,0] + cm[1,1])
print("\n3. SVM Accuracy = " + str(round(100*accuracy,2)))



""" Decision Tree Classifier """
# Fitting SVM to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

accuracy = (float(cm[0,0]) + cm[1,1])/(cm[0,0] + cm[0,1]+ cm[1,0] + cm[1,1])
print("\n3. Decision Tree Accuracy = " + str(round(100*accuracy,2)))



""" BernoulliRBM """
# Fitting SVM to the Training set
from sklearn.neural_network import BernoulliRBM
mlp = BernoulliRBM()
mlp.fit(X_train,y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

accuracy = (float(cm[0,0]) + cm[1,1])/(cm[0,0] + cm[0,1]+ cm[1,0] + cm[1,1])
print("\n4. BernoulliRBM Accuracy = " + str(round(100*accuracy,2)) + "\n\n")