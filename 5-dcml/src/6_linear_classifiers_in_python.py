# #--  --  --  -- Linear Classifiers in Python
# # Used for Data Scientist Training Path 
# #FYI it's a compilation of how to work
# #with different commands.


# ### --------------------------------------------------------
# # # # ------>>>> KNN classification
from sklearn.neighbors import KNeighborsClassifier

# Create and fit the model
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Predict on the test features, print the results
pred = knn.predict(X_test)[0]
print("Prediction for test example 0:", pred)


# ### --------------------------------------------------------
# # # # ------>>>> Comparing models
# Compare k nearest neighbors classifiers with k=1 
# and k=5 on the handwritten digits data set, which 
# is already loaded into the variables X_train, 
# y_train, X_test, and y_test. You can set k with 
# the n_neighbors parameter when creating the 
# KNeighborsClassifier object, which is also 
# already imported into the environment. 

# Which model has a higher test accuracy?
k1= KNeighborsClassifier(n_neighbors=1)
k5= KNeighborsClassifier(n_neighbors=5)

k1.fit(X_train, y_train)
k5.fit(X_train, y_train)

print(k1.score(X_test, y_test))
print(k5.score(X_test, y_test))
# R/ k = 5



# ### --------------------------------------------------------
# # # # ------>>>>  Overfitting
# Which of the following situations looks like an example of overfitting?
# R/ Training accuracy 95%, testing accuracy 50%.






# ### --------------------------------------------------------
# # # # ------>>>>  Running LogisticRegression and SVC
from sklearn import datasets
digits = datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

# Apply logistic regression and print scores
lr = LogisticRegression()
lr.fit(X_train, y_train)
print(lr.score(X_train, y_train))
print(lr.score(X_test, y_test))

# Apply SVM and print scores
svm = SVC()
svm.fit(X_train, y_train)
print(svm.score(X_train, y_train))
print(svm.score(X_test, y_test))




# ### --------------------------------------------------------
# # # # ------>>>>  Sentiment analysis for movie reviews
# Instantiate logistic regression and train
lr = LogisticRegression()
lr.fit(X, y)

# Predict sentiment for a glowing review
review1 = "LOVED IT! This movie was amazing. Top 10 this year."
review1_features = get_features(review1)
print("Review:", review1)
print("Probability of positive review:", lr.predict_proba(review1_features)[0,1])

# Predict sentiment for a poor review
review2 = "Total junk! I'll never watch a film by that director again, no matter how good the reviews."
review2_features = get_features(review2)
print("Review:", review2)
print("Probability of positive review:", lr.predict_proba(review2_features)[0,1])




# ### --------------------------------------------------------
# # # # ------>>>> Which decision boundary is linear?
# Which of the following is a linear decision boundary?
# R/ (1)



# ### --------------------------------------------------------
# # # # ------>>>> Visualizing decision boundaries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier

# Define the classifiers
classifiers = [LogisticRegression(), LinearSVC(),
               SVC(), KNeighborsClassifier()]

# Fit the classifiers
for c in classifiers:
    c.fit(X, y)

# Plot the classifiers
plot_4_classifiers(X, y, classifiers)
plt.show()




# ### --------------------------------------------------------
# # # # ------>>>> How models make predictions
# Which classifiers make predictions based on the sign (positive or negative) of the raw model output?
# R/ Both logistic regression and Linear SVMs
# Because --->  logistic regression and SVMs are both linear classifiers, the raw model output is a linear function of x.



# ### --------------------------------------------------------
# # # # ------>>>> Changing the model coefficients
# Set the coefficients
model.coef_ = np.array([[-1,1]])
model.intercept_ = np.array([-3])

# Plot the data and decision boundary
plot_classifier(X,y,model)

# Print the number of errors
num_err = np.sum(y != model.predict(X))
print("Number of errors:", num_err)




# ### --------------------------------------------------------
# # # # ------>>>> The 0-1 loss
# In the figure below, what is the 0-1 loss (number of classification errors) of the classifier?
# R/ 2 
# There is 1 misclassified red point and 1 misclassified blue point.



# ### --------------------------------------------------------
# # # # ------>>>> Minimizing a loss function
# The squared error, summed over training examples
def my_loss(w):
    s = 0
    for i in range(y.size):
        # Get the true and predicted target values for example 'i'
        y_i_true = y[i]
        y_i_pred = w@X[i]
        s = s + (y_i_true - y_i_pred)**2
    return s

# Returns the w that makes my_loss(w) smallest
w_fit = minimize(my_loss, X[0]).x
print(w_fit)

# Compare with scikit-learn's LinearRegression coefficients
lr = LinearRegression(fit_intercept=False).fit(X,y)
print(lr.coef_)




# ### --------------------------------------------------------
# # # # ------>>>> Classification loss functions
# Which of the four loss functions makes sense for classification?
# R/ (2)
# This loss is very similar to the hinge loss used in SVMs (just shifted slightly).



# ### --------------------------------------------------------
# # # # ------>>>> Comparing the logistic and hinge losses
# Mathematical functions for logistic and hinge losses
def log_loss(raw_model_output):
   return np.log(1+np.exp(-raw_model_output))
def hinge_loss(raw_model_output):
   return np.maximum(0,1-raw_model_output)

# Create a grid of values and plot
grid = np.linspace(-2,2,1000)
plt.plot(grid, log_loss(grid), label='logistic')
plt.plot(grid, hinge_loss(grid), label='hinge')
plt.legend()
plt.show()




# ### --------------------------------------------------------
# # # # ------>>>> Implementing logistic regression
# The logistic loss, summed over training examples
def my_loss(w):
    s = 0
    for i in range(y.size):
        raw_model_output = w@X[i]
        s = s + log_loss(raw_model_output * y[i])
    return s

# Returns the w that makes my_loss(w) smallest
w_fit = minimize(my_loss, X[0]).x
print(w_fit)

# Compare with scikit-learn's LogisticRegression
lr = LogisticRegression(fit_intercept=False, C=1000000).fit(X,y)
print(lr.coef_)




# ### --------------------------------------------------------
# # # # ------>>>> Regularized logistic regression
# Train and validaton errors initialized as empty list
train_errs = list()
valid_errs = list()

# Loop over values of C
for C_value in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    # Create LogisticRegression object and fit
    lr = LogisticRegression(C=C_value)
    lr.fit(X_train, y_train)

    # Evaluate error rates and append to lists
    train_errs.append(1.0 - lr.score(X_train, y_train))
    valid_errs.append(1.0 - lr.score(X_valid, y_valid))

# Plot results
plt.semilogx(C_values, train_errs, C_values, valid_errs)
plt.legend(("train", "validation"))
plt.show()




# ### --------------------------------------------------------
# # # # ------>>>> Logistic regression and feature selection
# Specify L1 regularization
lr = LogisticRegression(penalty='l1')

# Instantiate the GridSearchCV object and run the search
searcher = GridSearchCV(lr, {'C':[0.001, 0.01, 0.1, 1, 10]})
searcher.fit(X_train, y_train)

# Report the best parameters
print("Best CV params", searcher.best_params_)

# Find the number of nonzero coefficients (selected features)
best_lr = searcher.best_estimator_
coefs = best_lr.coef_
print("Total number of features:", coefs.size)
print("Number of selected features:", np.count_nonzero(coefs))





# ### --------------------------------------------------------
# # # # ------>>>> Identifying the most positive and negative words
# Get the indices of the sorted cofficients
inds_ascending = np.argsort(lr.coef_.flatten())
inds_descending = inds_ascending[::-1]

# Print the most positive words
print("Most positive words: ", end="")
for i in range(5):
    print(vocab[inds_descending[i]], end=", ")
print("\n")

# Print most negative words
print("Most negative words: ", end="")
for i in range(5):
    print(vocab[inds_ascending[i]], end=", ")
print("\n")




# ### --------------------------------------------------------
# # # # ------>>>> Getting class probabilities
# Which of the following transformations would make sense 
# for transforming the raw model output of a linear classifier
#  into a class probability?
# R/ (3)




# ### --------------------------------------------------------
# # # # ------>>>> Regularization and probabilities - part# 0
# Set the regularization strength
model = LogisticRegression(C=1)

# Fit and plot
model.fit(X,y)
plot_classifier(X,y,model,proba=True)

# Predict probabilities on training points
prob = model.predict_proba(X)
print("Maximum predicted probability", np.max(prob))

# # # # ------>>>> Regularization and probabilities - part# 1
# Set the regularization strength
model = LogisticRegression(C=0.1)

# Fit and plot
model.fit(X,y)
plot_classifier(X,y,model,proba=True)

# Predict probabilities on training points
prob = model.predict_proba(X)
print("Maximum predicted probability", np.max(prob))




# ### --------------------------------------------------------
# # # # ------>>>> Visualizing easy and difficult examples
lr = LogisticRegression()
lr.fit(X,y)

# Get predicted probabilities
proba = lr.predict_proba(X)

# Sort the example indices by their maximum probability
proba_inds = np.argsort(np.max(proba,axis=1))

# Show the most confident (least ambiguous) digit
show_digit(proba_inds[-1], lr)

# Show the least confident (most ambiguous) digit
show_digit(proba_inds[0], lr)




# ### --------------------------------------------------------
# # # # ------>>>> Counting the coefficients
# If you fit a logistic regression model on a classification problem with 3 classes and 100 features, how many 
# coefficients would you have, including intercepts?
# R/ 303


# ### --------------------------------------------------------
# # # # ------>>>> Fitting multi-class logistic regression
# Fit one-vs-rest logistic regression classifier
lr_ovr = LogisticRegression()
lr_ovr.fit(X_train, y_train)

print("OVR training accuracy:", lr_ovr.score(X_train, y_train))
print("OVR test accuracy    :", lr_ovr.score(X_test, y_test))

# Fit softmax classifier
lr_mn = LogisticRegression(multi_class="multinomial", solver="lbfgs")
lr_mn.fit(X_train, y_train)

print("Softmax training accuracy:", lr_mn.score(X_train, y_train))
print("Softmax test accuracy    :", lr_mn.score(X_test, y_test))



# ### --------------------------------------------------------
# # # # ------>>>> Visualizing multi-class logistic regression
# Print training accuracies
print("Softmax     training accuracy:", lr_mn.score(X_train, y_train))
print("One-vs-rest training accuracy:", lr_ovr.score(X_train, y_train))

# Create the binary classifier (class 1 vs. rest)
lr_class_1 = LogisticRegression(C=100)
lr_class_1.fit(X_train, y_train==1)

# Plot the binary classifier (class 1 vs. rest)
plot_classifier(X_train, y_train==1, lr_class_1)



# ### --------------------------------------------------------
# # # # ------>>>> One-vs-rest SVM
# We'll use SVC instead of LinearSVC from now on
from sklearn.svm import SVC

# Create/plot the binary classifier (class 1 vs. rest)
svm_class_1 = SVC()
svm_class_1.fit(X_train, y_train==1)
plot_classifier(X_train, y_train==1, svm_class_1)



# ### --------------------------------------------------------
# # # # ------>>>> Support vector definition
# Which of the following is a true statement about support
#  vectors? To help you out, here's the picture of support 
# vectors from the video (top), as well as the hinge loss 
# from Chapter 2 (bottom).
# R/ All incorrectly classified points are support vectors.





# ### --------------------------------------------------------
# # # # ------>>>> Effect of removing examples
# Train a linear SVM
svm = SVC(kernel="linear")
svm.fit(X,y)
plot_classifier(X, y, svm, lims=(11,15,0,6))

# Make a new data set keeping only the support vectors
print("Number of original examples", len(X))
print("Number of support vectors", len(svm.support_))
X_small = X[svm.support_]
y_small = y[svm.support_]

# Train a new SVM using only the support vectors
svm_small = SVC(kernel="linear")
svm_small.fit(X_small, y_small)
plot_classifier(X_small, y_small, svm_small, lims=(11,15,0,6))



# ### --------------------------------------------------------
# # # # ------>>>> GridSearchCV warm-up
# Instantiate an RBF SVM
svm = SVC()

# Instantiate the GridSearchCV object and run the search
parameters = {'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
searcher = GridSearchCV(svm, parameters)
searcher.fit(X, y)

# Report the best parameters
print("Best CV params", searcher.best_params_)





# ### --------------------------------------------------------
# # # # ------>>>> Jointly tuning gamma and C with GridSearchCV
# Instantiate an RBF SVM
svm = SVC()

# Instantiate the GridSearchCV object and run the search
parameters = {'C':[0.1, 1, 10], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
searcher = GridSearchCV(svm, parameters)
searcher.fit(X_train, y_train)

# Report the best parameters and the corresponding score
print("Best CV params", searcher.best_params_)
print("Best CV accuracy", searcher.best_score_)

# Report the test accuracy using these best parameters
print("Test accuracy of best grid search hypers:", searcher.score(X_test, y_test))





# ### --------------------------------------------------------
# # # # ------>>>> An advantage of SVMs
# Which of the following is an advantage of SVMs over logistic regression?
# R/ They are computationally efficient with kernels.






# ### --------------------------------------------------------
# # # # ------>>>> An advantage of logistic regression
# Which of the following is an advantage of logistic regression over SVMs?
# R/ It naturally outputs meaningful probabilities.






# ### --------------------------------------------------------
# # # # ------>>>> Using SGDClassifier
# We set random_state=0 for reproducibility
linear_classifier = SGDClassifier(random_state=0)

# Instantiate the GridSearchCV object and run the search
parameters = {'alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
             'loss':['hinge', 'log'], 'penalty':['l1','l2']}
searcher = GridSearchCV(linear_classifier, parameters, cv=10)
searcher.fit(X_train, y_train)

# Report the best parameters and the corresponding score
print("Best CV params", searcher.best_params_)
print("Best CV accuracy", searcher.best_score_)
print("Test accuracy of best grid search hypers:", searcher.score(X_test, y_test))

