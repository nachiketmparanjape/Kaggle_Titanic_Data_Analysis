#Predictive Analysis on Titanic Data

import pandas as pd
import numpy as np

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
        
#Make the gender more granular
titanic_df['person'] = titanic_df[['Age','Sex']].apply(male_female_child,axis=1)

#Alone or with family
titanic_df['Alone']=titanic_df.SibSp+titanic_df.Parch
titanic_df['Alone'].loc[titanic_df['Alone'] >0] = 1

df = titanic_df[['Survived','Pclass','Sex','Age','Alone','Embarked','Fare']]

#Dealing with NAs

# get average, std, and number of NaN values in titanic_df
average_age_titanic   = df["Age"].mean()
std_age_titanic       = df["Age"].std()
count_nan_age_titanic = df["Age"].isnull().sum()

#df = df.fillna(df.mean())
rand = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
df["Age"][np.isnan(df["Age"])] = rand
df = df.fillna(df.mean())

X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:, 1])
X[:, 4] = labelencoder.fit_transform(X[:, 4])
#onehotencoder = OneHotEncoder(categorical_features = [2])
#X = onehotencoder.fit_transform(X).toarray()

#Doing the same to test.csv

predict_df = pd.read_csv("test.csv")


#Alone or with family
predict_df['Alone']=predict_df.SibSp+predict_df.Parch
predict_df['Alone'].loc[predict_df['Alone'] >0] = 1

test_df = predict_df[['Pclass','Sex','Age','Alone','Embarked','Fare']]
#test_df = test_df.fillna(test_df.mean())

#Dealing with NAs

# get average, std, and number of NaN values in titanic_df
average_age_titanic   = test_df["Age"].mean()
std_age_titanic       = test_df["Age"].std()
count_nan_age_titanic = test_df["Age"].isnull().sum()

#df = df.fillna(df.mean())
rand = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
test_df["Age"][np.isnan(test_df["Age"])] = rand
test_df = test_df.fillna(test_df.mean())


Xtest = test_df.values
Xtest[:, 1] = labelencoder.fit_transform(Xtest[:, 1])
Xtest[:, 4] = labelencoder.fit_transform(Xtest[:, 1])



##############################
##############################

# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
#X_test2 = sc.transform(X_test)

#test.csv
Xtest2 = sc.transform(Xtest)


"""Naive Bayes"""

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X, y)

""" Naive Bayes output csv """

#Alone or with family
predict_df['Alone']=predict_df.SibSp+predict_df.Parch
predict_df['Alone'].loc[predict_df['Alone'] >0] = 1


test_df = test_df.fillna(test_df.mean())

y_pred2 = classifier.predict(Xtest2)

#Predictions
predict_df['Survived'] = y_pred2

#output csv
predict_df.to_csv('bayes_output.csv',columns=['PassengerId','Survived'],index=False)



"""Random Forest"""
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 116, criterion = 'entropy')
classifier.fit(X, y)

""" Random forest output csv """

#Alone or with family
predict_df['Alone']=predict_df.SibSp+predict_df.Parch
predict_df['Alone'].loc[predict_df['Alone'] >0] = 1


test_df = test_df.fillna(test_df.mean())

y_pred2 = classifier.predict(Xtest2)

#Predictions
predict_df['Survived'] = y_pred2

#output csv
predict_df.to_csv('forest_output2.csv',columns=['PassengerId','Survived'],index=False)



""" Support Vector Machines Classifier """
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0, gamma = 4)
classifier.fit(X, y)


""" SVM output csv """

#Alone or with family
predict_df['Alone']=predict_df.SibSp+predict_df.Parch
predict_df['Alone'].loc[predict_df['Alone'] >0] = 1


test_df = test_df.fillna(test_df.mean())

y_pred2 = classifier.predict(Xtest2)

#Predictions
predict_df['Survived'] = y_pred2

#output csv
predict_df.to_csv('svm_output.csv',columns=['PassengerId','Survived'],index=False)



""" Decision Tree Classifier """
# Fitting SVM to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X, y)

""" Decision Tree output csv """

#Alone or with family
predict_df['Alone']=predict_df.SibSp+predict_df.Parch
predict_df['Alone'].loc[predict_df['Alone'] >0] = 1


test_df = test_df.fillna(test_df.mean())

y_pred2 = classifier.predict(Xtest2)

#Predictions
predict_df['Survived'] = y_pred2

#output csv
predict_df.to_csv('tree_output.csv',columns=['PassengerId','Survived'],index=False)



""" BernoulliRBM """
# Fitting SVM to the Training set
from sklearn.neural_network import BernoulliRBM
mlp = BernoulliRBM()
mlp.fit(X,y)

""" BernoulliRBM output csv """

#Alone or with family
predict_df['Alone']=predict_df.SibSp+predict_df.Parch
predict_df['Alone'].loc[predict_df['Alone'] >0] = 1


test_df = test_df.fillna(test_df.mean())

y_pred2 = classifier.predict(Xtest2)

#Predictions
predict_df['Survived'] = y_pred2

#output csv
predict_df.to_csv('bernoulli_output.csv',columns=['PassengerId','Survived'],index=False)