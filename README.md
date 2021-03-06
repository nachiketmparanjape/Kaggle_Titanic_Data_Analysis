# A deep dive to see the survival patterns for the people who were on Titanic

This repo contains a Jupyter notebook with qualitative data analysis and visualization of the data available from the Titanic disaster.

## We try to find basic information such as -
1.) Who were the passengers on the Titanic? (Ages,Gender,Class,..etc)

2.) What deck were the passengers on and how does that relate to their class?

3.) Where did the passengers come from?

4.) Who was alone and who was with family?

### Descriptive Analysis

#### Age Distribuion
![age-histogram](https://cloud.githubusercontent.com/assets/11637437/23479727/51b70f40-fe7a-11e6-9b35-680f207f4c6d.png)

![age-kde-plot](https://cloud.githubusercontent.com/assets/11637437/23479730/51bf12f8-fe7a-11e6-9372-51ee2d862098.png)

#### Cabin Population
![cabin-population](https://cloud.githubusercontent.com/assets/11637437/23479731/51bfb42e-fe7a-11e6-8a63-c21bf46f5355.png)

#### Class
![gender-from-every-class](https://cloud.githubusercontent.com/assets/11637437/23479737/51d96b6c-fe7a-11e6-9d40-c628bddebd21.png)

![gender-from-every-class-children-included](https://cloud.githubusercontent.com/assets/11637437/23479738/51dc4d1e-fe7a-11e6-9e29-5a9cf5316a1a.png)

![gender-ratio-in-every-class](https://cloud.githubusercontent.com/assets/11637437/23479739/51ddf3f8-fe7a-11e6-88dd-a96d9419650d.png)

#### How many people were alone and how many were with family?
![alone-bar](https://cloud.githubusercontent.com/assets/11637437/23479729/51bebc90-fe7a-11e6-8210-3b4d60d596cf.png)

#### Where did people board from?
![boarding-point](https://cloud.githubusercontent.com/assets/11637437/23479732/51c168f0-fe7a-11e6-96fe-e77a1d6edac4.png)



## Then we try to dive a little deeper - what factors saved someone from sinking? (No pun intended)


#### 1) Overall Survival
![overall-survival](https://cloud.githubusercontent.com/assets/11637437/23479740/51dfadb0-fe7a-11e6-8f4e-298de952dbaa.png)


##### Effect of Class on Survival
![effect-of-class-on-survival](https://cloud.githubusercontent.com/assets/11637437/23479728/51be887e-fe7a-11e6-88d8-d91f5abefbe7.png)

#### How do people who were alone fare in this?
![family-survival](https://cloud.githubusercontent.com/assets/11637437/23479734/51ce78a6-fe7a-11e6-9a1a-4fa1ee43de2d.png)

Does it get nuanced with gender and age?

![gender-family-survival](https://cloud.githubusercontent.com/assets/11637437/23479736/51d47ff8-fe7a-11e6-942e-1da3d2cd5ed3.png)

![family-age-survival](https://cloud.githubusercontent.com/assets/11637437/23479733/51c95060-fe7a-11e6-87b1-ba097ddc7c37.png)

##### How does Survival change with age for different genders?
![Effect of age and gender on survival](https://cloud.githubusercontent.com/assets/11637437/23442028/f332e4c0-fdda-11e6-973c-a55eab59f2ba.png)

##### Class and Gender on Survival
![gender-class-survival](https://cloud.githubusercontent.com/assets/11637437/23479735/51cface4-fe7a-11e6-9878-2f578c386e88.png)


## Finally, several predictive algorithms are run on the data to identify the most effective feature and train the features to optimize the model performance. Here is the accuracy obtained with the following models (Cross-Validation was employed)

1. Logistic Regression Accuracy= 81.01

2. kNN Accuracy = 75.98

3. Naive Bayes Accuracy = 76.54

4. Random Forest Accuracy = 78.77

5. SVM Accuracy = 79.33

6. Decision Tree Accuracy = 81.01

7. BernoulliRBM Accuracy = 81.01

Done for now.

