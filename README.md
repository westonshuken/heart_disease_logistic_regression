# Making Predictions with Logistic Regression ([Scikit-Learn](https://scikit-learn.org/stable/))


In this tutorial, I will be sharing how to make predictions using logistic regression. The purpose of using logistic regression really begins with a question, that being a yes/no, true/false, or other binary question. In better terms, logistic regression is used for a binary classification problem and is considered to be a supervised machine learning/statistical model. 

An example of a scenario where logistic regression would be helpful could be when predicting if a patient has a disease or not, if an email is spam or not spam, or if a transaction is fraudulent or not fraudulent. 

To reiterate, logistic regression should not be used as a model to predict anything, but rather is a model used when a prediction is not continuous. For the sake of this tutorial, we will look at a binary classification problem, but logistic regression can be used to predict multiple classes of predictors. 

## Step 1
### Find the data and state the target

We will be using a dataset called *[Heart Disease UCI](https://www.kaggle.com/ronitf/heart-disease-uci)* from Kaggle. Specifically looking at the data from year 2019.

**Creator Credit:**
- Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
- University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
- University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
- V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.

The dataset had 13 predictor columns with information like age, sex, chest pain, cholesterol, and other heart-related metrics. The target column is binary (0,1) with 0 meaning no heart disease present and 1 meaning heart disease present. Our goal is to predict if someone has heart disease or not. To begin, let’s take a look at the dataset. 

## Step 2
### Do some exploratory data analysis

Import Pandas
```
import pandas as pd
```

Load the data
```
df = pd.read_csv('heart.csv')
```

View the data
```
df.head()
```
image

Check the datatypes of each column and check for null values
```
df.info()
```
image

Fortunately, there are no null values present and the categorical variables like *sex* have already been encoded

Check the decsriptive statistics 
```
round(df.describe(), 4)
```
image
With my lack of domain knowledge, I cannot tell too much here. However, I can see that there are binary values for the target, age ranges from 29 to 77, and there are more 1's than 0's in the target.

Taking a closer look at the target class balance
```
df['target'].value_counts()
```
image

It seems that the target class is quite balanced here, so not too much to worry about.

**Takeaways**: The dataset is very clean, but it has very few entries which is concerning especially when we split the data in the next step. Although the initial EDA seems a bit tedious, it is very important to do every time before modeling. 

## Step 3
### Train-Test Split

Here I will seperate the target from the predictor variables (y, X) and split those into training and testing sets. The default from Scikit-Learn is 25% on the testing data and 75% on the training data. I also created a random state for the purpose of the tutorial having the same results each time, feel free alter that state. As prior stated, we need to note of imbalance in the dataset since the amount of entries is quite small. Scikit will automatically balance the target variables, but imaging if something like age became unbalanced where older ages went into the test but not the train. Something to consider.


Code:
```
X = df.drop(['target'], axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=43214)
```
## Step 4
### Set up the pipeline

```
# Instantiate Standard Scaler and Logistic Regression
scaler = StandardScaler()
logreg = LogisticRegression(fit_intercept=False, C=1e16, solver='liblinear')

# Creating a classifier pipeline 
clf = Pipeline([('ss', scaler), 
                ('log', logreg)])
```

