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

I know ahead of time that I will be needing the scale my data, so I will begin with instantiating Scikit-Learn StandardScaler. The reason for this is because we want each predictor variable to be relative to one another. Therefore, the StandardScaler will "standardize features by removing the mean and scaling to unit variance" for each predictor using the z-score formula: **z = (x - u) / s**

*Important note*: the scaler should only be fit to the training set **not** the testing set in order to avoid data leakage.

After I intantiate the StandardScaler, I will instantiate Scikit-Learn LogisticRegression. I pass in the parameters fit_intercept=False, C=1e16, solver='liblinear'. The fit_intercept is set to True by default, and we do not need to add a constant in this case. The C value is the inverse regularization, which as of now is quite an arbitrary guess but can be adjusted later. The lower the number, the stronger the regularization. For the solver, I chose 'liblinear' because Scikit-Learn recommends for small datasets.

Finally, added the StandardScaler and LogisticRegression objects into a pipeline which helps for minimal code and making adjustments.

Code:
```
# Instantiate Standard Scaler and Logistic Regression
scaler = StandardScaler()
logreg = LogisticRegression(fit_intercept=False, C=1e16, solver='liblinear')

# Creating a classifier pipeline 
clf = Pipeline([('ss', scaler), 
                ('log', logreg)])
```

## Step 5
### Fit the model
Here we want to simply fit the model to our **training data**, then we will be ready to examine performance metrics.

Code:
```
clf.fit(X_train, y_train)
```

## Step 6
### Examine accuracy and log loss scores

The first metrics I will look at are the accuracy and log loss scores. 

The accuracy score will tell us the rate that the model correctly predicted the positive and negative results (true positives + true negatives / total).

The log loss scores will show the error for each model. Similary to MSE or RMSE in a linear a linear model, but in this case it is a bit harder to interpret, but in general a lower log-loss value means better predictions. Its main use here is to compare the train and the test performance. 

Code:
```
# Get the train and test accuracy scores
print(f"Train Score: {clf.score(X_train, y_train)}")
print(f"Test Score: {clf.score(X_test, y_test)}")

Train Score: 0.8722466960352423
Test Score: 0.8157894736842105
```

Code:
```
# Get the train and test logloss results
print(f"Train LogLoss: {log_loss(y_train, clf.predict_proba(X_train))}")
print(f"Test LogLoss: {log_loss(y_test, clf.predict_proba(X_test))}")

Train LogLoss: 0.35896912360998906
Test LogLoss: 0.36611952610146714
```
What do these mettrics mean?

## Step 7
### Confusion Matrix

Here we will visualize a confusion matrix of our testing results. The confusion matrix will show the the true positive, true negative, false postive, and false negative results.

Code:
```
plot_confusion_matrix(clf, X=X_test, y_true=y_test, display_labels=['No Disease', 'Disease'])
plt.grid(False);
```
image

We can easily cross check our accuracy measure here: (true neg + true pos) / total = 62/76 = 81%

Moreover, we can see where the model failed. In this case there are both false positives and false negatives (which we don't want). In this case we can assume that false negatives are the least desirable. This is given the cost of each. If someone is told they are negative, but actually are positive, this risks their life. on the other hand, if someone is told they are positive when they are negative, a further test could shed light on the truth.

This situation brings us to our next set of metrics: recall, precision, and F1-score.


## Step 8
### Precision, Recall, & F1-score
