# #--  --  --  -- Supervised Learning with scikit-learn
# # Used for Data Scientist Training Path 
# #FYI it's a compilation of how to work
# #with different commands.


# ### --------------------------------------------------------
# # # # ------>>>> Which of these is a 
# classification problem? Once 
# you decide to leverage 
# supervised machine learning 
# to solve a new problem, you 
# need to identify whether your 
# problem is better suited to 
# classification or regression. 
# This exercise will help you 
# develop your intuition for 
# distinguishing between the 
# two. 

# Provided below are 4 example 
# applications of machine 
# learning. Which of them is a 
# supervised classification 
# problem?
# R/ Using labeled financial data to predict whether the value of a stock will go up or go down next week.


# ### --------------------------------------------------------
# # # # ------>>>>  Numerical EDA
# In this chapter, you'll be 
# working with a dataset 
# obtained from the UCI Machine 
# Learning Repository 
# consisting of votes made by 
# US House of Representatives 
# Congressmen. Your goal will 
# be to predict their party 
# affiliation ('Democrat' or 
# 'Republican') based on how 
# they voted on certain key 
# issues. Here, it's worth 
# noting that we have 
# preprocessed this dataset to 
# deal with missing values. 
# This is so that your focus 
# can be directed towards 
# understanding how to train 
# and evaluate supervised 
# learning models. Once you 
# have mastered these 
# fundamentals, you will be 
# introduced to preprocessing 
# techniques in Chapter 4 and 
# have the chance to apply them 
# there yourself - including on 
# this very same dataset! 

# Before thinking about what 
# supervised learning models 
# you can apply to this, 
# however, you need to perform 
# Exploratory data analysis (
# EDA) in order to understand 
# the structure of the data. 
# For a refresher on the 
# importance of EDA, check out 
# the first two chapters of 
# Statistical Thinking in 
# Python (Part 1). 

# Get started with your EDA now 
# by exploring this voting 
# records dataset numerically. 
# It has been pre-loaded for 
# you into a DataFrame called 
# df. Use pandas' .head(), 
# .info(), and .describe() 
# methods in the IPython Shell 
# to explore the DataFrame, and 
# select the statement below 
# that is not true.
df.head()
df.info()
df.describe()
# R/There are 17 predictor variables, or features, in this DataFrame.


# ### --------------------------------------------------------
# # # # ------>>>> Visual EDA
# The Numerical EDA you did in 
# the previous exercise gave 
# you some very important 
# information, such as the 
# names and data types of the 
# columns, and the dimensions 
# of the DataFrame. Following 
# this with some visual EDA 
# will give you an even better 
# understanding of the data. In 
# the video, Hugo used the 
# scatter_matrix() function on 
# the Iris data for this 
# purpose. However, you may 
# have noticed in the previous 
# exercise that all the 
# features in this dataset are 
# binary; that is, they are 
# either 0 or 1. So a different 
# type of plot would be more 
# useful here, such as 
# Seaborn's countplot. 

# Given on the right is a 
# countplot of the 'education' 
# bill, generated from the 
# following code: 

# plt.figure() sns.countplot(
# x='education', hue='party', 
# data=df, palette='RdBu') 
# plt.xticks([0,1], ['No', 
# 'Yes']) plt.show() In 
# sns.countplot(), we specify 
# the x-axis data to be 
# 'education', and hue to be 
# 'party'. Recall that 'party' 
# is also our target variable. 
# So the resulting plot shows 
# the difference in voting 
# behavior between the two 
# parties for the 'education' 
# bill, with each party colored 
# differently. We manually 
# specified the color to be 
# 'RdBu', as the Republican 
# party has been traditionally 
# associated with red, and the 
# Democratic party with blue. 

# It seems like Democrats voted 
# resoundingly against this 
# bill, compared to 
# Republicans. This is the kind 
# of information that our 
# machine learning model will 
# seek to learn when we try to 
# predict party affiliation 
# solely based on voting 
# behavior. An expert in U.S 
# politics may be able to 
# predict this without machine 
# learning, but probably not 
# instantaneously - and 
# certainly not if we are 
# dealing with hundreds of 
# samples! 

# In the IPython Shell, explore 
# the voting behavior further 
# by generating countplots for 
# the 'satellite' and 'missile' 
# bills, and answer the 
# following question: Of these 
# two bills, for which ones do 
# Democrats vote resoundingly 
# in favor of, compared to 
# Republicans? Be sure to begin 
# your plotting statements for 
# each figure with plt.figure() 
# so that a new figure will be 
# set up. Otherwise, your plots 
# will be overlaid onto the 
# same figure.
# R/ Both 'satellite' and 'missile'.



# ### --------------------------------------------------------
# # # # ------>>>>  k-Nearest Neighbors: Fit
# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier
# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values
# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)
# Fit the classifier to the data
knn.fit(X,y)



# ### --------------------------------------------------------
# # # # ------>>>> k-Nearest Neighbors: Predict
# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier 
# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values
# Create a k-NN classifier with 6 neighbors: knn
knn =  knn = KNeighborsClassifier(n_neighbors=6)
# Fit the classifier to the data
knn.fit(X,y)
# Predict the labels for the training data X
y_pred = knn.predict(X)
# Predict and print the label for the new data point X_new
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction))




# ### --------------------------------------------------------
# # # # ------>>>> The digits recognition dataset
# Import necessary modules
from sklearn import datasets
import matplotlib.pyplot as plt
# Load the digits dataset: digits
digits = datasets.load_digits()
# Print the keys and DESCR of the dataset
print(digits.keys())
print(digits.DESCR)
# Print the shape of the images and data keys
print(digits.images.shape)
print(digits.data.shape)
# Display digit 1010
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()



# ### --------------------------------------------------------
# # # # ------>>>> Train/Test Split + Fit/Predict/Accuracy
# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
# Create feature and target arrays
X = digits.data
y = digits.target
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)
# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)
# Fit the classifier to the training data
knn.fit(X_train, y_train)
# Print the accuracy
print(knn.score(X_test, y_test))



# ### --------------------------------------------------------
# # # # ------>>>> Overfitting and underfitting
# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit the classifier to the training data
    knn.fit(X_train, y_train)    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)
# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()




# ### --------------------------------------------------------
# # # # ------>>>> Which of the following is a 
# regression problem? Andy 
# introduced regression to you 
# using the Boston housing 
# dataset. But regression 
# models can be used in a 
# variety of contexts to solve 
# a variety of different 
# problems. 

# Given below are four example 
# applications of machine 
# learning. Your job is to pick 
# the one that is best framed 
# as a regression problem
# R/ A bike share company using time and weather data to predict the number of bikes being rented at any given hour.



# ### --------------------------------------------------------
# # # # ------>>>> Importing data for supervised learning
# Import numpy and pandas
import numpy as np
import pandas as pd
# Read the CSV file into a DataFrame: df
df = pd.read_csv('gapminder.csv')
# Create arrays for features and target variable
y = df['life'].values
X = df['fertility'].values
# Print the dimensions of X and y before reshaping
print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))
# Reshape X and y
y = y.reshape(-1,1)
X = X.reshape(-1,1)
# Print the dimensions of X and y after reshaping
print("Dimensions of y after reshaping: {}".format(y.shape))
print("Dimensions of X after reshaping: {}".format(X.shape))




# ### --------------------------------------------------------
# # # # ------>>>> Exploring the Gapminder data
# As always, it is important to 
# explore your data before 
# building models. On the 
# right, we have constructed a 
# heatmap showing the 
# correlation between the 
# different features of the 
# Gapminder dataset, which has 
# been pre-loaded into a 
# DataFrame as df and is 
# available for exploration in 
# the IPython Shell. Cells that 
# are in green show positive 
# correlation, while cells that 
# are in red show negative 
# correlation. Take a moment to 
# explore this: Which features 
# are positively correlated 
# with life, and which ones are 
# negatively correlated? Does 
# this match your intuition? 

# Then, in the IPython Shell, 
# explore the DataFrame using 
# pandas methods such as 
# .info(), .describe(), .head()
# . 

# In case you are curious, the 
# heatmap was generated using 
# Seaborn's heatmap function 
# and the following line of 
# code, where df.corr() 
# computes the pairwise 
# correlation between columns: 

# sns.heatmap(df.corr(), 
# square=True, cmap='RdYlGn') 

# Once you have a feel for the 
# data, consider the statements 
# below and select the one that 
# is not true. After this, Hugo 
# will explain the mechanics of 
# linear regression in the next 
# video and you will be on your 
# way building regression 
# models!
df.info()
df.head()
# R/fertility is of type int64.



# ### --------------------------------------------------------
# # # # ------>>>> Fit & predict for regression
# Import LinearRegression
from sklearn.linear_model import LinearRegression
# Create the regressor: reg
reg = LinearRegression()
# Create the prediction space
prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)
# Fit the model to the data
reg.fit(X_fertility, y)
# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)
# Print R^2 
print(reg.score(X_fertility, y))
# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()


# ### --------------------------------------------------------
# # # # ------>>>> Train/test split for regression
# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
# Create the regressor: reg_all
reg_all = LinearRegression()
# Fit the regressor to the training data
reg_all.fit(X_train, y_train)
# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)
# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))




# ### --------------------------------------------------------
# # # # ------>>>> 5-fold cross-validation
# Import the necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
# Create a linear regression object: reg
reg = LinearRegression()
# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg,X,y,cv=5)
# Print the 5-fold cross-validation scores
print(cv_scores)
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))




# ### --------------------------------------------------------
# # # # ------>>>> K-Fold CV comparison
# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
# Create a linear regression object: reg
reg = LinearRegression()
# Perform 3-fold CV
cvscores_3 = cross_val_score(reg, X, y, cv=3)
print(np.mean(cvscores_3))
# Perform 10-fold CV
cvscores_10 = cross_val_score(reg, X, y, cv=10)
print(np.mean(cvscores_10))




# ### --------------------------------------------------------
# # # # ------>>>> Regularization I: Lasso
# Import Lasso
from sklearn.linear_model import Lasso
# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.4, normalize=True)
# Fit the regressor to the data
lasso.fit(X,y)
# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)
# Plot the coefficients
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()




# ### --------------------------------------------------------
# # # # ------>>>> Regularization II: Ridge
# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []
# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)
# Compute scores over range of alphas
for alpha in alpha_space:
    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha    
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)    
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))  
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))
# Display the plot
display_plot(ridge_scores, ridge_scores_std)




# ### --------------------------------------------------------
# # # # ------>>>> Metrics for classification
# Import necessary modules
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=42)
# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier(n_neighbors=6)
# Fit the classifier to the training data
knn.fit(X_train, y_train)
# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)
# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



# ### --------------------------------------------------------
# # # # ------>>>> Building a logistic regression model
# Import the necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)
# Create the classifier: logreg
logreg = LogisticRegression()
# Fit the classifier to the training data
logreg.fit(X_train, y_train)
# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)
# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



# ### --------------------------------------------------------
# # # # ------>>>> Plotting an ROC curve
# Import necessary modules
from sklearn.metrics import roc_curve
# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]
# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()



# # ### --------------------------------------------------------
# # # # # ------>>>> Precision-recall Curve
# When looking at your ROC 
# curve, you may have noticed 
# that the y-axis (True 
# positive rate) is also known 
# as recall. Indeed, in 
# addition to the ROC curve, 
# there are other ways to 
# visually evaluate model 
# performance. One such way is 
# the precision-recall curve, 
# which is generated by 
# plotting the precision and 
# recall for different 
# thresholds. On the right, a 
# precision-recall curve has 
# been generated for the 
# diabetes dataset. The 
# classification report and 
# confusion matrix are 
# displayed in the IPython 
# Shell. 

# Study the precision-recall 
# curve and then consider the 
# statements given below. 
# Choose the one statement that 
# is not true. Note that here, 
# the class is positive (1) if 
# # the individual has diabetes.
# R/ -> Precision and recall take true negatives into consideration.
#  ---> negatives do not appear at all in the definitions of precision and recall.



# ### --------------------------------------------------------
# # # # ------>>>>AUC computation
# Import necessary modules
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]
# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))
# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(logreg, X, y, cv=5,
scoring='roc_auc')
# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))


# ### --------------------------------------------------------
# # # # ------>>>> Hyperparameter tuning with GridSearchCV
# Import necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}
# Instantiate a logistic regression classifier: logreg
logreg = LogisticRegression()
# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)
# Fit it to the data
logreg_cv.fit(X, y)
# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
print("Best score is {}".format(logreg_cv.best_score_))



# ### --------------------------------------------------------
# # # # ------>>>>Hyperparameter tuning with RandomizedSearchCV
# Import necessary modules
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))



# ### --------------------------------------------------------
# # # # ------>>>>-Hold-out set reasoning
# For which of the following reasons would you want to use a hold-out set for the very end?
# R/ You want to be absolutely certain about your model's ability to generalize to unseen data.



# ### --------------------------------------------------------
# # # # ------>>>> Hold-out set in practice I: Classification
# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Create the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression()

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the training data
logreg_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))



# ### --------------------------------------------------------
# # # # ------>>>> Hold-out set in practice II: Regression
# Import necessary modules
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 42)

# Create the hyperparameter grid
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space}

# Instantiate the ElasticNet regressor: elastic_net
elastic_net = ElasticNet()

# Setup the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)

# Fit it to the training data
gm_cv.fit(X_train, y_train)

# Predict on the test set and compute metrics
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))



# ### --------------------------------------------------------
# # # # ------>>>> Exploring categorical features
# Import pandas
import pandas as pd

# Read 'gapminder.csv' into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create a boxplot of life expectancy per region
df.boxplot('life', 'Region', rot=60)

# Show the plot
plt.show()



# ### --------------------------------------------------------
# # # # ------>>>> Creating dummy variables
# Create dummy variables: df_region
df_region = pd.get_dummies(df)

# Print the columns of df_region
print(df_region.columns)

# Create dummy variables with drop_first=True: df_region
df_region = pd.get_dummies(df, drop_first=True)

# Print the new columns of df_region
print(df_region.columns)



# ### --------------------------------------------------------
# # # # ------>>>> Regression with categorical features
# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Instantiate a ridge regressor: ridge
ridge = Ridge(normalize=True, alpha=0.5)

# Perform 5-fold cross-validation: ridge_cv
ridge_cv = cross_val_score(ridge, X, y, cv=5)

# Print the cross-validated scores
print(ridge_cv)



# ### --------------------------------------------------------
# # # # ------>>>> Dropping missing data
# Convert '?' to NaN
df[df == '?'] = np.nan

# Print the number of NaNs
print(df.isnull().sum())

# Print shape of original DataFrame
print("Shape of Original DataFrame: {}".format(df.shape))

# Drop missing values and print shape of new DataFrame
df = df.dropna()

# Print shape of new DataFrame
print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))



# ### --------------------------------------------------------
# # # # ------>>>> Imputing missing data in a ML Pipeline I
# Import the Imputer module
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC

# Setup the Imputation transformer: imp
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

# Instantiate the SVC classifier: clf
clf = SVC()

# Setup the pipeline with the required steps: steps
steps = [('imputation', imp),
        ('SVM', clf)]



# ### --------------------------------------------------------
# # # # ------>>>> Imputing missing data in a ML Pipeline II
# Import necessary modules
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
        ('SVM', SVC())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the train set
pipeline.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))



# ### --------------------------------------------------------
# # # # ------>>>> Centering and scaling your data
# Import scale
from sklearn.preprocessing import scale

# Scale the features: X_scaled
X_scaled = scale(X)

# Print the mean and standard deviation of the unscaled features
print("Mean of Unscaled Features: {}".format(np.mean(X))) 
print("Standard Deviation of Unscaled Features: {}".format(np.std(X)))

# Print the mean and standard deviation of the scaled features
print("Mean of Scaled Features: {}".format(np.mean(X_scaled))) 
print("Standard Deviation of Scaled Features: {}".format(np.std(X_scaled)))



# ### --------------------------------------------------------
# # # # ------>>>> Centering and scaling in a pipeline
# Import the necessary modules
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())]
        
# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the training set: knn_scaled
knn_scaled = pipeline.fit(X_train, y_train)

# Instantiate and fit a k-NN classifier to the unscaled data
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)

# Compute and print metrics
print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test, y_test)))
print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test, y_test)))



# ### --------------------------------------------------------
# # # # ------>>>> Bringing it all together I: Pipeline for classification
# Setup the pipeline
steps = [('scaler', StandardScaler()),
         ('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=21)

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline, param_grid=parameters)

# Fit to the training set
cv.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))



# ### --------------------------------------------------------
# # # # ------>>>> Bringing it all together II: Pipeline for regression
 # Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
         ('scaler', StandardScaler()),
         ('elasticnet', ElasticNet())]

# Create the pipeline: pipeline 
pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(pipeline, param_grid=parameters)

# Fit to the training set
gm_cv.fit(X_train, y_train)

# Compute and print the metrics
r2 = gm_cv.score(X_test, y_test)
print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))



