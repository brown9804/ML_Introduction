  
# #--  --  --  --  Case Study: School Budgeting with Machine Learning in Python:
# # Used for Data Scientist Training Path 
# #FYI it's a compilation of how to work
# #with different commands.


# ### --------------------------------------------------------
# # # # ------>>>> What category of problem is this?
# You're no novice to data science, but 
# let's make sure we agree on the basics. 
# As Peter from DrivenData explained in 
# the video, you're going to be working 
# with school district budget data. This 
# data can be classified in many ways 
# according to certain labels, e.g. 
# Function: Career & Academic Counseling, 
# or Position_Type: Librarian. Your goal 
# is to develop a model that predicts the 
# probability for each possible label by 
# relying on some correctly labeled 
# examples. What type of machine learning 
# problem is this?
# R/  Supervised Learning, because the model will be trained using labeled examples.



# ### --------------------------------------------------------
# # # # ------>>>> What is the goal of the algorithm?
# As you know from previous courses, 
# there are different types of supervised 
# machine learning problems. In this 
# exercise you will tell us what type of 
# supervised machine learning problem 
# this is, and why you think so. 
# Remember, your goal is to correctly 
# label budget line items by training a 
# supervised model to predict the 
# probability of each possible label, 
# taking most probable label as the 
# correct label.
# R/ Classification, because predicted probabilities will be used to select a label class.



# # ### --------------------------------------------------------
# # # # # ------>>>> Loading the data
# Now it's time to check out the dataset! 
# You'll use pandas (which has been pre-
# imported as pd) to load your data into 
# a DataFrame and then do some 
# Exploratory Data Analysis (EDA) of it. 
# The training data is available as 
# TrainingData.csv. Your first task is to 
# load it into a DataFrame in the IPython 
# Shell using pd.read_csv() along with 
# the keyword argument index_col=0. Use 
# methods such as .info(), .head(), and 
# .tail() to explore the budget data and 
# the properties of the features and 
# labels. Some of the column names 
# correspond to features - descriptions 
# of the budget items - such as the 
# Job_Title_Description column. The 
# values in this column tell us if a 
# budget item is for a teacher, 
# custodian, or other employee. Some 
# columns correspond to the budget item 
# labels you will be trying to predict 
# with your model. For example, the 
# Object_Type column describes whether 
# the budget item is related classroom 
# supplies, salary, travel expenses, etc. 
# Use df.info() in the IPython Shell to 
# answer the following questions: How 
# many rows are there in the training 
# data? How many columns are there in the 
# training data? How many non-null 
# entries are in the 
# Job_Title_Description column?
pd.read_csv('TrainingData.csv', index_col=0)
# R/ 1560 rows, 25 columns, 1131 non-null entries in Job_Title_Description.




# ### --------------------------------------------------------
# # # # ------>>>> Summarizing the data
# Print the summary statistics
print(df.describe())

# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Create the histogram
plt.hist(df['FTE'].dropna())

# Add title and labels
plt.title('Distribution of %full-time \n employee works')
plt.xlabel('% of full-time')
plt.ylabel('num employees')

# Display the histogram
plt.show()




# ### --------------------------------------------------------
# # # # ------>>>> Exploring datatypes in pandas
# It's always good to know what datatypes 
# you're working with, especially when 
# the inefficient pandas type object may 
# be involved. Towards that end, let's 
# explore what we have. The data has been 
# loaded into the workspace as df. Your 
# job is to look at the DataFrame 
# attribute .dtypes in the IPython Shell, 
# and call its .value_counts() method in 
# order to answer the question below. 
# Make sure to call 
# df.dtypes.value_counts(), and not 
# df.value_counts()! Check out the 
# difference in the Shell. 
# df.value_counts() will return an error, 
# because it is a Series method, not a 
# DataFrame method. How many columns with 
# dtype object are in the data?
df.dtypes.value_counts()
# R/ 23



# ### --------------------------------------------------------
# # # # ------>>>> Encode the labels as categorical variables
# Define the lambda function: categorize_label
categorize_label = lambda x: x.astype('category')

# Convert df[LABELS] to a categorical type
df[LABELS] = df[LABELS].apply(categorize_label, axis=0)

# Print the converted dtypes
print(df[LABELS].dtypes)




# ### --------------------------------------------------------
# # # # ------>>>> Counting unique labels
# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Calculate number of unique values for each label: num_unique_labels
num_unique_labels = df[LABELS].apply(pd.Series.nunique)

# Plot number of unique values for each label
num_unique_labels.plot(kind='bar')

# Label the axes
plt.xlabel('Labels')
plt.ylabel('Number of unique values')

# Display the plot
plt.show()




# ### --------------------------------------------------------
# # # # ------>>>> Penalizing highly confident wrong answers
# As Peter explained in the video, log loss provides a steep penalty
#  for predictions that are both wrong and confident, i.e., a high probability 
# is assigned to the incorrect class.
# Select the ordering of the examples which corresponds to the lowest
#  to highest log loss scores. y is an indicator of whether the example 
# was classified correctly. You shouldn't need to crunch any numbers!
# R/ Lowest: A, Middle: C, Highest: B.



# ### --------------------------------------------------------
# # # # ------>>>> Computing log loss with NumPy
# Compute and print log loss for 1st case
correct_confident_loss = compute_log_loss(correct_confident, actual_labels)
print("Log loss, correct and confident: {}".format(correct_confident_loss)) 

# Compute log loss for 2nd case
correct_not_confident_loss = compute_log_loss(correct_not_confident, actual_labels)
print("Log loss, correct and not confident: {}".format(correct_not_confident_loss)) 

# Compute and print log loss for 3rd case
wrong_not_confident_loss = compute_log_loss(wrong_not_confident, actual_labels)
print("Log loss, wrong and not confident: {}".format(wrong_not_confident_loss)) 

# Compute and print log loss for 4th case
wrong_confident_loss = compute_log_loss(wrong_confident, actual_labels)
print("Log loss, wrong and confident: {}".format(wrong_confident_loss)) 

# Compute and print log loss for actual labels
actual_labels_loss = compute_log_loss(actual_labels, actual_labels)
print("Log loss, actual labels: {}".format(actual_labels_loss)) 
 




# ### --------------------------------------------------------
# # # # ------>>>> Setting up a train-test split in scikit-learn
# Create the new DataFrame: numeric_data_only
numeric_data_only = df[NUMERIC_COLUMNS].fillna(-1000)

# Get labels and convert to dummy variables: label_dummies
label_dummies = pd.get_dummies(df[LABELS])

# Create training and test sets
X_train, X_test, y_train, y_test = multilabel_train_test_split(numeric_data_only,label_dummies,size=0.2,seed=123)

# Print the info
print("X_train info:")
print(X_train.info())
print("\nX_test info:")  
print(X_test.info())
print("\ny_train info:")  
print(y_train.info())
print("\ny_test info:")  
print(y_test.info()) 




# ### --------------------------------------------------------
# # # # ------>>>> Training a model
# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Create the DataFrame: numeric_data_only
numeric_data_only = df[NUMERIC_COLUMNS].fillna(-1000)

# Get labels and convert to dummy variables: label_dummies
label_dummies = pd.get_dummies(df[LABELS])

# Create training and test sets
X_train, X_test, y_train, y_test = multilabel_train_test_split(numeric_data_only,
                                                               label_dummies,
                                                               size=0.2, 
                                                               seed=123)

# Instantiate the classifier: clf
clf = OneVsRestClassifier(LogisticRegression())

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Print the accuracy
print("Accuracy: {}".format(clf.score(X_test, y_test)))




# ### --------------------------------------------------------
# # # # ------>>>> Use your model to predict values on holdout data
# Instantiate the classifier: clf
clf = OneVsRestClassifier(LogisticRegression())

# Fit it to the training data
clf.fit(X_train, y_train)

# Load the holdout data: holdout
holdout = pd.read_csv('HoldoutData.csv', index_col=0)

# Generate predictions: predictions
predictions = clf.predict_proba(holdout[NUMERIC_COLUMNS].fillna(-1000))




# ### --------------------------------------------------------
# # # # ------>>>> Writing out your results to a csv for submission
# Generate predictions: predictions
predictions = clf.predict_proba(holdout[NUMERIC_COLUMNS].fillna(-1000))

# Format predictions in DataFrame: prediction_df
prediction_df = pd.DataFrame(columns=pd.get_dummies(df[LABELS]).columns,
                             index=holdout.index,
                             data=predictions)


# Save prediction_df to csv
prediction_df.to_csv('predictions.csv')

# Submit the predictions for scoring: score
score = score_submission(pred_path='predictions.csv')

# Print score
print('Your model, trained with numeric data only, yields logloss score: {}'.format(score))




# ### --------------------------------------------------------
# # # # ------>>>> Tokenizing text
# As we talked about in the video, 
# tokenization is the process of chopping 
# up a character sequence into pieces 
# called tokens. How do we determine what 
# constitutes a token? Often, tokens are 
# separated by whitespace. But we can 
# specify other delimiters as well. For 
# example, if we decided to tokenize on 
# punctuation, then any punctuation mark 
# would be treated like a whitespace. How 
# we tokenize text in our DataFrame can 
# affect the statistics we use in our 
# model. A particular cell in our budget 
# DataFrame may have the string content 
# Title I - Disadvantaged 
# Children/Targeted Assistance. The 
# number of n-grams generated by this 
# text data is sensitive to whether or 
# not we tokenize on punctuation, as 
# you'll show in the following exercise. 
# How many tokens (1-grams) are in the 
# string Title I - Disadvantaged 
# Children/Targeted Assistance if we 
# tokenize on whitespace and punctuation?
# R/ 6 
# Because -> Tokenizing on whitespace and punctuation means that Children/Targeted becomes two tokens and
#  - is dropped altogether. Nice work!



# ### --------------------------------------------------------
# # # # ------>>>> Testing your NLP credentials with n-grams
# You're well on your way to NLP 
# superiority. Let's test your mastery of 
# n-grams! In the workspace, we have the 
# loaded a python list, one_grams, which 
# contains all 1-grams of the string 
# petro-vend fuel and fluids, tokenized 
# on punctuation. Specifically, one_grams 
# = ['petro', 'vend', 'fuel', 'and', 
# 'fluids'] In this exercise, your job is 
# to determine the sum of the sizes of 1-
# grams, 2-grams and 3-grams generated by 
# the string petro-vend fuel and fluids, 
# tokenized on punctuation. Recall that 
# the n-gram of a sequence consists of 
# all ordered subsequences of length n.
# 1-grams + 2-grams + 3-grams is 5 + 4 + 3 = 12
# R/ 12 


# ### --------------------------------------------------------
# # # # ------>>>> Creating a bag-of-words in scikit-learn
# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Fill missing values in df.Position_Extra
df.Position_Extra.fillna('', inplace=True)

# Instantiate the CountVectorizer: vec_alphanumeric
vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

# Fit to the data
vec_alphanumeric.fit(df.Position_Extra)

# Print the number of tokens and first 15 tokens
msg = "There are {} tokens in Position_Extra if we split on non-alpha numeric"
print(msg.format(len(vec_alphanumeric.get_feature_names())))
print(vec_alphanumeric.get_feature_names()[:15])




# ### --------------------------------------------------------
# # # # ------>>>> Combining text columns for tokenization
# Define combine_text_columns()
def combine_text_columns(data_frame, to_drop=NUMERIC_COLUMNS + LABELS):
    """ converts all text in each row of data_frame to single vector """
    
    # Drop non-text columns that are in the df
    to_drop = set(to_drop) & set(data_frame.columns.tolist())
    text_data = data_frame.drop(to_drop, axis=1)
    
    # Replace nans with blanks
    text_data.fillna("", inplace=True)
    
    # Join all text items in a row that have a space in between
    return text_data.apply(lambda x: " ".join(x), axis=1)




# ### --------------------------------------------------------
# # # # ------>>>> What's in a token?
# Import the CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create the basic token pattern
TOKENS_BASIC = '\\S+(?=\\s+)'

# Create the alphanumeric token pattern
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Instantiate basic CountVectorizer: vec_basic
vec_basic = CountVectorizer(token_pattern=TOKENS_BASIC)

# Instantiate alphanumeric CountVectorizer: vec_alphanumeric
vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

# Create the text vector
text_vector = combine_text_columns(df)

# Fit and transform vec_basic
vec_basic.fit_transform(text_vector)

# Print number of tokens of vec_basic
print("There are {} tokens in the dataset".format(len(vec_basic.get_feature_names())))

# Fit and transform vec_alphanumeric
vec_alphanumeric.fit_transform(text_vector)

# Print number of tokens of vec_alphanumeric
print("There are {} alpha-numeric tokens in the dataset".format(len(vec_alphanumeric.get_feature_names())))




# ### --------------------------------------------------------
# # # # ------>>>> Instantiate pipeline
# Import Pipeline
from sklearn.pipeline import Pipeline

# Import other necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Split and select numeric data only, no nans 
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric']],
                                                    pd.get_dummies(sample_df['label']), 
                                                    random_state=22)

# Instantiate Pipeline object: pl
pl = Pipeline([
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

# Fit the pipeline to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - numeric, no nans: ", accuracy)




# ### --------------------------------------------------------
# # # # ------>>>> Preprocessing numeric features
# Import the Imputer object
from sklearn.preprocessing import Imputer

# Create training and test sets using only numeric data
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric', 'with_missing']], pd.get_dummies(sample_df['label']), random_state=456)

# Insantiate Pipeline object: pl
pl = Pipeline([
        ('imp', Imputer()),
        ('clf', OneVsRestClassifier(LogisticRegression()))])

# Fit the pipeline to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - all numeric, incl nans: ", accuracy)



# ### --------------------------------------------------------
# # # # ------>>>> Preprocessing text features
# Import the CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Split out only the text data
X_train, X_test, y_train, y_test = train_test_split(sample_df['text'],
                                                    pd.get_dummies(sample_df['label']), 
                                                    random_state=456)

# Instantiate Pipeline object: pl
pl = Pipeline([
        ('vec', CountVectorizer()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - just text data: ", accuracy)



# ### --------------------------------------------------------
# # # # ------>>>> Multiple types of processing: FunctionTransformer
# Import FunctionTransformer
from sklearn.preprocessing import FunctionTransformer

# Obtain the text data: get_text_data
get_text_data = FunctionTransformer(lambda x: x['text'], validate=False)

# Obtain the numeric data: get_numeric_data
get_numeric_data = FunctionTransformer(lambda x: x[['numeric', 'with_missing']], validate=False)

# Fit and transform the text data: just_text_data
just_text_data = get_text_data.fit_transform(sample_df)

# Fit and transform the numeric data: just_numeric_data
just_numeric_data = get_numeric_data.fit_transform(sample_df)

# Print head to check results
print('Text Data')
print(just_text_data.head())
print('\nNumeric Data')
print(just_numeric_data.head())



# ### --------------------------------------------------------
# # # # ------>>>> Multiple types of processing: FeatureUnion
# Import FeatureUnion
from sklearn.pipeline import FeatureUnion

# Split using ALL data in sample_df
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric', 'with_missing', 'text']],
                                                    pd.get_dummies(sample_df['label']), 
                                                    random_state=22)

# Create a FeatureUnion with nested pipeline: process_and_join_features
process_and_join_features = FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )

# Instantiate nested pipeline: pl
pl = Pipeline([
        ('union', process_and_join_features),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])


# Fit pl to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - all data: ", accuracy)



# ### --------------------------------------------------------
# # # # ------>>>> Using FunctionTransformer on the main dataset
# Import FunctionTransformer
from sklearn.preprocessing import FunctionTransformer

# Get the dummy encoding of the labels
dummy_labels = pd.get_dummies(df[LABELS])

# Get the columns that are features in the original df
NON_LABELS = [c for c in df.columns if c not in LABELS]

# Split into training and test sets
X_train, X_test, y_train, y_test = multilabel_train_test_split(df[NON_LABELS],
                                                               dummy_labels,
                                                               0.2, 
                                                               seed=123)

# Preprocess the text data: get_text_data
get_text_data = FunctionTransformer(combine_text_columns, validate=False)

# Preprocess the numeric data: get_numeric_data
get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS], validate=False)



# ### --------------------------------------------------------
# # # # ------>>>> Add a model to the pipeline
# Complete the pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)



# ### --------------------------------------------------------
# # # # ------>>>> Try a different class of model
# Import random forest classifer
from sklearn.ensemble import RandomForestClassifier

# Edit model step in pipeline
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )),
        ('clf', RandomForestClassifier())
    ])

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)



# ### --------------------------------------------------------
# # # # ------>>>> Can you adjust the model or parameters to improve accuracy?
# Import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# Add model step to pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )),
        ('clf', RandomForestClassifier(n_estimators=15))
    ])

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)



# ### --------------------------------------------------------
# # # # ------>>>> How many tokens?
# Recall from previous chapters 
# that how you tokenize text 
# affects the n-gram statistics 
# used in your model. Going 
# forward, you'll use alpha-
# numeric sequences, and only 
# alpha-numeric sequences, as 
# tokens. Alpha-numeric tokens 
# contain only letters a-z and 
# numbers 0-9 (no other 
# characters). In other words, 
# you'll tokenize on 
# punctuation to generate n-
# gram statistics. In this 
# exercise, you'll make sure 
# you remember how to tokenize 
# on punctuation. Assuming we 
# tokenize on punctuation, 
# accepting only alpha-numeric 
# sequences as tokens, how many 
# tokens are in the following 
# string from the main dataset? 
# 'PLANNING,RES,DEV,& EVAL      
# ' If you want, we've loaded 
# this string into the 
# workspace as SAMPLE_STRING, 
# but you may not need it to 
# answer the question.
# R/ 4, because , and & are not tokens
# Because ->
# Commas, "&", and whitespace
#  are not alpha-numeric tokens. Keep it up!


# ### --------------------------------------------------------
# # # # ------>>>> Deciding what's a word
# Import the CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create the text vector
text_vector = combine_text_columns(X_train)

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Instantiate the CountVectorizer: text_features
text_features = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

# Fit text_features to the text vector
text_features.fit(text_vector)

# Print the first 10 tokens
print(text_features.get_feature_names()[:10])



# ### --------------------------------------------------------
# # # # ------>>>> N-gram range in scikit-learn
# Import pipeline
from sklearn.pipeline import Pipeline

# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Import other preprocessing modules
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import chi2, SelectKBest

# Select 300 best features
chi_k = 300

# Import functional utilities
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from sklearn.pipeline import FeatureUnion

# Perform preprocessing
get_text_data = FunctionTransformer(combine_text_columns, validate=False)
get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS], validate=False)

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Instantiate pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                   ngram_range=(1, 2))),
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
             ]
        )),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])



# ### --------------------------------------------------------
# # # # ------>>>> Which models of the data include interaction terms?
# Recall from the video that interaction terms involve products of features.
# Suppose we have two features x and y, and we use models that 
# process the features as follows:
# βx + βy + ββ
# βxy + βx + βy
# βx + βy + βx^2 + βy^2
# where β is a coefficient in your model (not a feature).

# Which expression(s) include interaction terms?

# R/ The second expression.
# Because ->
# An xy term is present, which represents interactions between features. 
# Nice work, let''s implement this!


# ### --------------------------------------------------------
# # # # ------>>>> Implement interaction modeling in scikit-learn
# Instantiate pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                   ngram_range=(1, 2))),  
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
             ]
        )),
        ('int', SparseInteractions(degree=2)),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])



# ### --------------------------------------------------------
# # # # ------>>>> Why is hashing a useful trick?
# In the video, Peter explained that a hash 
# function takes an input, in your case a token, 
# and outputs a hash value. For example, the input 
# may be a string and the hash value may be an 
# integer. We've loaded a familiar python datatype, 
# a dictionary called hash_dict, that makes this 
# mapping concept a bit more explicit. In fact, 
# python dictionaries ARE hash tables! Print 
# hash_dict in the IPython Shell to get a sense of 
# how strings can be mapped to integers. By 
# explicitly stating how many possible outputs the 
# hashing function may have, we limit the size of 
# the objects that need to be processed. With these 
# limits known, computation can be made more 
# efficient and we can get results faster, even on 
# large datasets. Using the above information, 
# answer the following: Why is hashing a useful 
# trick?
hash_dict
# R/ Some problems are memory-bound and not easily parallelizable, 
# and hashing enforces a fixed length computation instead of using a 
# mutable datatype (like a dictionary).
# Because:  Enforcing a fixed length can speed up calculations drastically, especially on large datasets!




# ### --------------------------------------------------------
# # # # ------>>>> Implementing the hashing trick in scikit-learn
# Import HashingVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

# Get text data: text_data
text_data = combine_text_columns(X_train)

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)' 

# Instantiate the HashingVectorizer: hashing_vec
hashing_vec = HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

# Fit and transform the Hashing Vectorizer
hashed_text = hashing_vec.fit_transform(text_data)

# Create DataFrame and print the head
hashed_df = pd.DataFrame(hashed_text.data)
print(hashed_df.head())




# ### --------------------------------------------------------
# # # # ------>>>> Build the winning model
# Import the hashing vectorizer
from sklearn.feature_extraction.text import HashingVectorizer

# Instantiate the winning model pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                     non_negative=True, norm=None, binary=False,
                                                     ngram_range=(1,2))),
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
             ]
        )),
        ('int', SparseInteractions(degree=2)),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])




# ### --------------------------------------------------------
# # # # ------>>>> What tactics got the winner the best score?
# Now you've implemented the winning model from 
# start to finish. If you want to use this model 
# locally, this Jupyter notebook contains all the 
# code you've worked so hard on. You can now take 
# that code and build on it! Let's take a moment to 
# reflect on why this model did so well. What 
# tactics got the winner the best score?
# R/ The winner used skillful NLP, efficient computation, and simple but powerful stats tricks to master the budget data.
# Because -> Often times simpler is better, and understanding the problem in depth leads to simpler solutions!