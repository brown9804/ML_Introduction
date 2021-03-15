  
# #--  --  --  --  Extreme Gradient Boosting with XGBoost:
# # Used for Data Scientist Training Path 
# #FYI it's a compilation of how to work
# #with different commands.


# ### --------------------------------------------------------
# # # # ------>>>> Which of these is a classification problem?
# Given below are 4 potential machine learning
# # problems you might encounter in the wild.
# # Pick the one that is a classification problem.
# Given past performance of stocks and various other financial data, predicting the exact price of a given stock (Google) tomorrow.
# Given a large dataset of user behaviors on a website, generating an informative segmentation of the users based on their behaviors.
# Predicting whether a given user will click on an ad given the ad content and metadata associated with the user.
# Given a user's past behavior on a video platform, presenting him/her with a series of recommended videos to watch next.
# # R/Predicting whether a given user will click on an ad given the ad content and metadata associated with the user.



# ### --------------------------------------------------------
# # # # ------>>>> Which of these is a binary classification problem?
# Great! A classification problem involves 
# predicting the category a given data point 
# belongs to out of a finite set of possible 
# categories. Depending on how many possible 
# categories there are to predict, a classification 
# problem can be either binary or multi-class. 
# Let's do another quick refresher here. Your job 
# is to pick the binary classification problem out 
# of the following list of supervised learning 
# problems.
# Predicting whether a given image contains a cat.
# Predicting the emotional valence of a sentence (Valence can be positive, negative, or neutral).
# Recommending the most tax-efficient strategy for tax filing in an automated accounting system.
# Given a list of symptoms, generating a rank-ordered list of most likely diseases.
# R/ Predicting whether a given image contains a cat.



# ### --------------------------------------------------------
# # # # ------>>>> XGBoost: Fit/Predict
# Import xgboost
import xgboost as xgb

# Create arrays for the features and the target: X, y
X, y = churn_data.iloc[:,:-1], churn_data.iloc[:,-1]

# Create the training and test sets
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=123)

# Instantiate the XGBClassifier: xg_cl
xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123)

# Fit the classifier to the training set
xg_cl.fit(X_train, y_train)

# Predict the labels of the test set: preds
preds = xg_cl.predict(X_test)

# Compute the accuracy: accuracy
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))





# ### --------------------------------------------------------
# # # # ------>>>> Decision trees
# Import the necessary modules
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

# Create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Instantiate the classifier: dt_clf_4
dt_clf_4 = DecisionTreeClassifier(max_depth=4)

# Fit the classifier to the training set
dt_clf_4.fit(X_train, y_train) 

# Predict the labels of the test set: y_pred_4
y_pred_4 =  dt_clf_4.predict(X_test)

# Compute the accuracy of the predictions: accuracy
accuracy = float(np.sum(y_pred_4==y_test))/y_test.shape[0]
print("accuracy:", accuracy)





# ### --------------------------------------------------------
# # # # ------>>>> Measuring accuracy
# Create arrays for the features and the target: X, y
X, y = churn_data.iloc[:,:-1], churn_data.iloc[:,-1]

# Create the DMatrix from X and y: churn_dmatrix
churn_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:logistic", "max_depth":3}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, 
                  nfold=3, num_boost_round=5, 
                  metrics="error", as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Print the accuracy
print(((1-cv_results["test-error-mean"]).iloc[-1]))




# ### --------------------------------------------------------
# # # # ------>>>> Measuring AUC
# Perform cross_validation: cv_results
cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, 
                  nfold=3, num_boost_round=5, 
                  metrics="auc", as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Print the AUC
print((cv_results["test-auc-mean"]).iloc[-1])



# ### --------------------------------------------------------
# # # # ------>>>> Using XGBoost
# XGBoost is a powerful library that scales very 
# well to many samples and works for a variety of 
# supervised learning problems. That said, as 
# Sergey described in the video, you shouldn't 
# always pick it as your default machine learning 
# library when starting a new project, since there 
# are some situations in which it is not the best 
# option. In this exercise, your job is to consider 
# the below examples and select the one which would 
# be the best use of XGBoost.
# Visualizing the similarity between stocks by comparing the time series of their historical prices relative to each other.
# Predicting whether a person will develop cancer using genetic data with millions of genes, 23 examples of genomes of people that didn't develop cancer, 3 genomes of people who wound up getting cancer.
# Clustering documents into topics based on the terms used in them.
# Predicting the likelihood that a given user will click an ad from a very large clickstream log with millions of users and their web interactions.
# R/ Predicting the likelihood that a given user will click an ad from a very large clickstream log with millions of users and their web interactions.





# ### --------------------------------------------------------
# # # # ------>>>> Which of these is a regression problem?
# Here are 4 potential machine learning problems 
# # you might encounter in the wild. Pick the one that 
# # is a clear example of a regression problem.
# Recommending a restaurant to a user given their past history of restaurant visits and reviews for a dining aggregator app.
# Predicting which of several thousand diseases a given person is most likely to have given their symptoms.
# Tagging an email as spam/not spam based on its content and metadata (sender, time sent, etc.).
# Predicting the expected payout of an auto insurance claim given claim properties (car, accident type, driver prior history, etc.).
# R/ # Predicting the expected payout of an auto insurance claim given claim properties (car, accident type, driver prior history, etc.).




# ### --------------------------------------------------------
# # # # ------>>>> Decision trees as base learners
# Create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Instantiate the XGBRegressor: xg_reg
xg_reg = xgb.XGBRegressor(objective='reg:linear', n_estimators=10, seed=123)

# Fit the regressor to the training set
xg_reg.fit(X_train,y_train)

# Predict the labels of the test set: preds
preds = xg_reg.predict(X_test)

# Compute the rmse: rmse
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))




# ### --------------------------------------------------------
# # # # ------>>>> Linear base learners
# Convert the training and testing sets into DMatrixes: DM_train, DM_test
DM_train = xgb.DMatrix(data=X_train, label=y_train)
DM_test =  xgb.DMatrix(data=X_test, label=y_test)

# Create the parameter dictionary: params
params = {"booster":"gblinear", "objective":"reg:linear"}

# Train the model: xg_reg
xg_reg = xgb.train(params = params, dtrain=DM_train, num_boost_round=5)

# Predict the labels of the test set: preds
preds = xg_reg.predict(DM_test)

# Compute and print the RMSE
rmse = np.sqrt(mean_squared_error(y_test,preds))
print("RMSE: %f" % (rmse))




# ### --------------------------------------------------------
# # # # ------>>>> Evaluating model quality - Part#0
# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:linear", "max_depth":4}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=4, num_boost_round=5, metrics='rmse', as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Extract and print final boosting round metric
print((cv_results["test-rmse-mean"]).tail(1))

# # # # ------>>>> Evaluating model quality - Part#1
# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:linear", "max_depth":4}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=4, num_boost_round=5, metrics='mae', as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Extract and print final boosting round metric
print((cv_results["test-mae-mean"]).tail(1))




# ### --------------------------------------------------------
# # # # ------>>>> Using regularization in XGBoost
# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

reg_params = [1, 10, 100]

# Create the initial parameter dictionary for varying l2 strength: params
params = {"objective":"reg:linear","max_depth":3}

# Create an empty list for storing rmses as a function of l2 complexity
rmses_l2 = []

# Iterate over reg_params
for reg in reg_params:

    # Update l2 strength
    params["lambda"] = reg
    
    # Pass this updated param dictionary into cv
    cv_results_rmse = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=2, num_boost_round=5, metrics="rmse", as_pandas=True, seed=123)
    
    # Append best rmse (final round) to rmses_l2
    rmses_l2.append(cv_results_rmse["test-rmse-mean"].tail(1).values[0])

# Look at best rmse per l2 param
print("Best rmse as a function of l2:")
print(pd.DataFrame(list(zip(reg_params, rmses_l2)), columns=["l2", "rmse"]))




# ### --------------------------------------------------------
# # # # ------>>>> Visualizing individual XGBoost trees
# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:linear", "max_depth":2}

# Train the model: xg_reg
xg_reg = xgb.train(params=params, dtrain=housing_dmatrix, num_boost_round=10)

# Plot the first tree
xgb.plot_tree(xg_reg,num_trees=0)
plt.show()

# Plot the fifth tree
xgb.plot_tree(xg_reg,num_trees=4)
plt.show()

# Plot the last tree sideways
xgb.plot_tree(xg_reg,num_trees=9, rankdir="LR")
plt.show()




# ### --------------------------------------------------------
# # # # ------>>>> Visualizing feature importances: What features are most important in my dataset
# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X,label=y)

# Create the parameter dictionary: params
params = {'objective':'reg:linear', 'max_depth':4}

# Train the model: xg_reg
xg_reg = xgb.train(dtrain=housing_dmatrix, params=params, num_boost_round=10)

# Plot the feature importances
xgb.plot_importance(xg_reg)
plt.show()




# ### --------------------------------------------------------
# # # # ------>>>> When is tuning your model a bad idea?
# Now that you've seen the 
# effect that tuning has on the 
# overall performance of your 
# XGBoost model, let's turn the 
# question on its head and see 
# if you can figure out when 
# tuning your model might not 
# be the best idea. Given that 
# model tuning can be time-
# intensive and complicated, 
# which of the following 
# scenarios would NOT call for 
# careful tuning of your model?
# You have lots of examples from some dataset and very many features at your disposal.

# You are very short on time before you must push an initial model to production and have little data to train your model on.

# You have access to a multi-core (64 cores) server with lots of memory (200GB RAM) and no time constraints.

# You must squeeze out every last bit of performance out of your xgboost model.

# R/ You are very short on time before you must push an initial model to production and have little data to train your model on.




# ### --------------------------------------------------------
# # # # ------>>>> Tuning the number of boosting rounds
# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary for each tree: params 
params = {"objective":"reg:linear", "max_depth":3}

# Create list of number of boosting rounds
num_rounds = [5, 10, 15]

# Empty list to store final round rmse per XGBoost model
final_rmse_per_round = []

# Iterate over num_rounds and build one model per num_boost_round parameter
for curr_num_rounds in num_rounds:

    # Perform cross-validation: cv_results
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3, num_boost_round=curr_num_rounds, metrics="rmse", as_pandas=True, seed=123)
    
    # Append final round RMSE
    final_rmse_per_round.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
num_rounds_rmses = list(zip(num_rounds, final_rmse_per_round))
print(pd.DataFrame(num_rounds_rmses,columns=["num_boosting_rounds","rmse"]))




# ### --------------------------------------------------------
# # # # ------>>>> Automated boosting round selection using early_stopping
# Create your housing DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary for each tree: params
params = {"objective":"reg:linear", "max_depth":4}

# Perform cross-validation with early stopping: cv_results
cv_results = xgb.cv(dtrain=housing_dmatrix, params= params, num_boost_round=50, nfold=3, metrics = 'rmse', early_stopping_rounds=10, as_pandas=True, seed=123)

# Print cv_results
print(cv_results)




# ### --------------------------------------------------------
# # # # ------>>>> Tuning eta
# Create your housing DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary for each tree (boosting round)
params = {"objective":"reg:linear", "max_depth":3}

# Create list of eta values and empty list to store final round rmse per xgboost model
eta_vals = [0.001, 0.01, 0.1]
best_rmse = []

# Systematically vary the eta 
for curr_val in eta_vals:

    params["eta"] = curr_val
    
    # Perform cross-validation: cv_results
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params,nfold=3,num_boost_round=10,early_stopping_rounds=5,metrics='rmse',as_pandas=True, seed=123)
    
    
    
    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
print(pd.DataFrame(list(zip(eta_vals, best_rmse)), columns=["eta","best_rmse"]))


# ### --------------------------------------------------------
# # # # ------>>>> Tuning max_depth
# Create your housing DMatrix
housing_dmatrix = xgb.DMatrix(data=X,label=y)

# Create the parameter dictionary
params = {"objective":"reg:linear"}

# Create list of max_depth values
max_depths = [2,5,10,20]
best_rmse = []

# Systematically vary the max_depth
for curr_val in max_depths:

    params["max_depth"] = curr_val
    
    # Perform cross-validation
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, metrics='rmse', num_boost_round=10, early_stopping_rounds=5,nfold=2,as_pandas=True,seed=123)
    
    
    
    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
print(pd.DataFrame(list(zip(max_depths, best_rmse)),columns=["max_depth","best_rmse"]))


# ### --------------------------------------------------------
# # # # ------>>>> Tuning colsample_bytree
# Create your housing DMatrix
housing_dmatrix = xgb.DMatrix(data=X,label=y)

# Create the parameter dictionary
params={"objective":"reg:linear","max_depth":3}

# Create list of hyperparameter values: colsample_bytree_vals
colsample_bytree_vals = [0.1,0.5,0.8,1]
best_rmse = []

# Systematically vary the hyperparameter value 
for curr_val in colsample_bytree_vals:

    params['colsample_bytree'] = curr_val
    
    # Perform cross-validation
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=2,
                 num_boost_round=10, early_stopping_rounds=5,
                 metrics="rmse", as_pandas=True, seed=123)
    
    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
print(pd.DataFrame(list(zip(colsample_bytree_vals, best_rmse)), columns=["colsample_bytree","best_rmse"]))

# ### --------------------------------------------------------
# # # # ------>>>> Grid search with XGBoost
# Create the parameter grid: gbm_param_grid
gbm_param_grid = {
    'colsample_bytree': [0.3, 0.7],
    'n_estimators': [50],
    'max_depth': [2, 5]
}

# Instantiate the regressor: gbm
gbm = xgb.XGBRegressor()

# Perform grid search: grid_mse
grid_mse = GridSearchCV(estimator=gbm, param_grid=gbm_param_grid,cv=4,scoring='neg_mean_squared_error', verbose=1)


# Fit grid_mse to the data
grid_mse.fit(X,y)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", grid_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))


# ### --------------------------------------------------------
# # # # ------>>>> Random search with XGBoost
# Create the parameter grid: gbm_param_grid 
gbm_param_grid = {
    'n_estimators': [25],
    'max_depth': range(2, 12)
}

# Instantiate the regressor: gbm
gbm = xgb.XGBRegressor(n_estimators=10)

# Perform random search: grid_mse
randomized_mse = RandomizedSearchCV(estimator=gbm,cv=4,n_iter=5, verbose=1, param_distributions =gbm_param_grid,scoring='neg_mean_squared_error')


# Fit randomized_mse to the data
randomized_mse.fit(X,y)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", randomized_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(randomized_mse.best_score_)))



# ### --------------------------------------------------------
# # # # ------>>>> When should you use grid search and random search?
# Now that you've seen some of the drawbacks of grid search and random 
# search, which of the following most accurately describes why both random 
# search and grid search are non-ideal search hyperparameter tuning 
# strategies in all scenarios?
# R/ The search space size can be massive for Grid Search in certain cases, whereas for Random Search the number of hyperparameters has a significant effect on how long it takes to run.



# ### --------------------------------------------------------
# # # # ------>>>> Exploratory data analysis
# Before diving into the nitty 
# gritty of pipelines and 
# preprocessing, let's do some 
# exploratory analysis of the 
# original, unprocessed Ames 
# housing dataset. When you 
# worked with this data in 
# previous chapters, we 
# preprocessed it for you so 
# you could focus on the core 
# XGBoost concepts. In this 
# chapter, you'll do the 
# preprocessing yourself! 

# A smaller version of this 
# original, unprocessed dataset 
# has been pre-loaded into a 
# pandas DataFrame called df. 
# Your task is to explore df in 
# the Shell and pick the option 
# that is incorrect. The larger 
# purpose of this exercise is 
# to understand the kinds of 
# transformations you will need 
# to perform in order to be 
# able to use XGBoost.
df
df.head()
df.info()
df.isnull()
# R/ The LotFrontage column has no missing values and its entries are of type float64.
# Because ->
# The LotFrontage column actually does
#  have missing values: 259, to be precise. 
#  Additionally, notice how columns such as
#   MSZoning, PavedDrive, and HouseStyle are categorical.
#    These need to be encoded numerically before you can
#     use XGBoost. This is what you'll do in the coming exercises.


# ### --------------------------------------------------------
# # # # ------>>>> Encoding categorical columns I: LabelEncoder
# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Fill missing values with 0
df.LotFrontage = df.LotFrontage.fillna(0)

# Create a boolean mask for categorical columns
categorical_mask = (df.dtypes == 'object')

# Get list of categorical column names
categorical_columns = df.columns[categorical_mask].tolist()

# Print the head of the categorical columns
print(df[categorical_columns].head())

# Create LabelEncoder object: le
le = LabelEncoder()

# Apply LabelEncoder to categorical columns
df[categorical_columns] = df[categorical_columns].apply(lambda x: le.fit_transform(x))

# Print the head of the LabelEncoded categorical columns
print(df[categorical_columns].head())


# ### --------------------------------------------------------
# # # # ------>>>> Encoding categorical columns II: OneHotEncoder
# Import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

# Create OneHotEncoder: ohe
ohe = OneHotEncoder(categorical_features=categorical_mask,sparse=False)

# Apply OneHotEncoder to categorical columns - output is no longer a dataframe: df_encoded
df_encoded = ohe.fit_transform(df)

# Print first 5 rows of the resulting dataset - again, this will no longer be a pandas dataframe
print(df_encoded[:5, :])

# Print the shape of the original DataFrame
print(df.shape)

# Print the shape of the transformed array
print(df_encoded.shape)


# ### --------------------------------------------------------
# # # # ------>>>> Encoding categorical columns III: DictVectorizer
# Import DictVectorizer
from sklearn.feature_extraction import DictVectorizer

# Convert df into a dictionary: df_dict
df_dict = df.to_dict('records')

# Create the DictVectorizer object: dv
dv = DictVectorizer(sparse=False)

# Apply dv on df: df_encoded
df_encoded = dv.fit_transform(df_dict)

# Print the resulting first five rows
print(df_encoded[:5,:])

# Print the vocabulary
print(dv.vocabulary_)


# ### --------------------------------------------------------
# # # # ------>>>> Preprocessing within a pipeline
# Import necessary modules
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer

# Fill LotFrontage missing values with 0
X.LotFrontage = X.LotFrontage.fillna(0)

# Setup the pipeline steps: steps
steps = [("ohe_onestep", DictVectorizer(sparse=False)),
         ("xgb_model", xgb.XGBRegressor())]

# Create the pipeline: xgb_pipeline
xgb_pipeline = Pipeline(steps)

# Fit the pipeline
xgb_pipeline.fit(X.to_dict('records'),y)



# ### --------------------------------------------------------
# # # # ------>>>> Cross-validating your XGBoost model
# Import necessary modules
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Fill LotFrontage missing values with 0
X.LotFrontage = X.LotFrontage.fillna(0)

# Setup the pipeline steps: steps
steps = [("ohe_onestep", DictVectorizer(sparse=False)),
         ("xgb_model", xgb.XGBRegressor(max_depth=2, objective="reg:linear"))]

# Create the pipeline: xgb_pipeline
xgb_pipeline = Pipeline(steps)

# Cross-validate the model
cross_val_scores = cross_val_score(xgb_pipeline, X.to_dict('records'), y, cv=10, scoring='neg_mean_squared_error')

# Print the 10-fold RMSE
print("10-fold RMSE: ", np.mean(np.sqrt(np.abs(cross_val_scores))))




# ### --------------------------------------------------------
# # # # ------>>>>  Kidney disease case study I: Categorical Imputer
# Import necessary modules
from sklearn_pandas import DataFrameMapper
from sklearn_pandas import CategoricalImputer

# Check number of nulls in each feature column
nulls_per_column = X.isnull().sum()
print(nulls_per_column)

# Create a boolean mask for categorical columns
categorical_feature_mask = X.dtypes == object

# Get list of categorical column names
categorical_columns = X.columns[categorical_feature_mask].tolist()

# Get list of non-categorical column names
non_categorical_columns = X.columns[~categorical_feature_mask].tolist()

# Apply numeric imputer
numeric_imputation_mapper = DataFrameMapper(
                                            [([numeric_feature], Imputer(strategy="median")) for numeric_feature in non_categorical_columns],
                                            input_df=True,
                                            df_out=True
                                           )

# Apply categorical imputer
categorical_imputation_mapper = DataFrameMapper(
                                                [(category_feature, CategoricalImputer()) for category_feature in categorical_columns],
                                                input_df=True,
                                                df_out=True
                                               )






# ### --------------------------------------------------------
# # # # ------>>>> Kidney disease case study II: Feature Union
# Import FeatureUnion
from sklearn.pipeline import FeatureUnion

# Combine the numeric and categorical transformations
numeric_categorical_union = FeatureUnion([
                                          ("num_mapper", numeric_imputation_mapper),
                                          ("cat_mapper", categorical_imputation_mapper)
                                         ])




# ### --------------------------------------------------------
# # # # ------>>>> Kidney disease case study III: Full pipeline
# Create full pipeline
pipeline = Pipeline([
                     ("featureunion", numeric_categorical_union),
                     ("dictifier", Dictifier()),
                     ("vectorizer", DictVectorizer(sort=False)),
                     ("clf", xgb.XGBClassifier(max_depth=3))
                    ])

# Perform cross-validation
cross_val_scores = cross_val_score(pipeline, X, y, scoring="roc_auc", cv=3)

# Print avg. AUC
print("3-fold AUC: ", np.mean(cross_val_scores))



# ### --------------------------------------------------------
# # # # ------>>>>  Bringing it all together
# Create the parameter grid
gbm_param_grid = {
    'clf__learning_rate': np.arange(0.05, 1, 0.05),
    'clf__max_depth': np.arange(3, 10, 1),
    'clf__n_estimators': np.arange(50, 200, 50)
}

# Perform RandomizedSearchCV
randomized_roc_auc = RandomizedSearchCV(estimator=pipeline,param_distributions=gbm_param_grid,scoring='roc_auc',cv=2,n_iter=2,verbose=1)

# Fit the estimator
randomized_roc_auc.fit(X,y)

# Compute metrics
print(randomized_roc_auc.best_score_)
print(randomized_roc_auc.best_estimator_)