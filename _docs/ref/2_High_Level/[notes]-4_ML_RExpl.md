# [Notes] Machine Learning Results/Explainability
----------

Costa Rica

Belinda Brown, belindabrownr04@gmail.com

Jan, 2021

----------

## `Display Results`
Based on [2], and [3]:
``` python 
from azureml.widgets import RunDetails
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

# Show best run details 
RunDetails(best_run).show()

# Create Confusion Matrix 
#### ------ Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC(kernel='linear', C=0.01).fit(X_train, y_train)
np.set_printoptions(precision=2)
#### ------ Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
  disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=categorical_columns,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
  disp.ax_.set_title(title)

  print(title)
  print(disp.confusion_matrix)
```

Based on [4]:
```python 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline

# Create the estimator - pipeline
pipeline = make_pipeline(StandardScaler(), LogisticRegression(random_state=1))
# Create training test splits using two features
pipeline.fit(X_train[:,[2, 13]],y_train)
probs = pipeline.predict_proba(X_test[:,[2, 13]])
fpr1, tpr1, thresholds = roc_curve(y_test, probs[:, 1], pos_label=1)
roc_auc1 = auc(fpr1, tpr1)
# Create training test splits using two different features
pipeline.fit(X_train[:,[4, 14]],y_train)
probs2 = pipeline.predict_proba(X_test[:,[4, 14]])
fpr2, tpr2, thresholds = roc_curve(y_test, probs2[:, 1], pos_label=1)
roc_auc2 = auc(fpr2, tpr2)
# Create training test splits using all features
pipeline.fit(X_train,y_train)
probs3 = pipeline.predict_proba(X_test)
# ROC Curve 
fpr3, tpr3, thresholds = roc_curve(y_test, probs3[:, 1], pos_label=1)
# AUC
roc_auc3 = auc(fpr3, tpr3)

fig, ax = plt.subplots(figsize=(7.5, 7.5))
plt.plot(fpr1, tpr1, label='ROC Curve 1 (AUC = %0.2f)' % (roc_auc1))
plt.plot(fpr2, tpr2, label='ROC Curve 2 (AUC = %0.2f)' % (roc_auc2))
plt.plot(fpr3, tpr3, label='ROC Curve 3 (AUC = %0.2f)' % (roc_auc3))
plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Random Classifier')   
plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', color='green', label='Perfect Classifier')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc="lower right")
plt.show()
```
Considering:
> `predict()` is used to predict the actual class (In your case one of 0, 1 or 1).
> `predict_proba()` is used to predict the class probabilities
> As an example:
> `predict()` would output class 0 since the class probability for 0 is 0.6.
> [0.6, 0.2, 0.2] is the output of `predict_proba` that simply denotes that the class probability 
> for classes 0, 1 and 2 are 0.6, 0.2 and 0.2 respectively.
>
> > -- <cite> Stack Overflow from [5] </cite>

``` python 
X_validation = validation_data.drop_columns(columns=target_column).to_pandas_dataframe()
y_validation = validation_data.keep_columns(columns=target_column, validate=True).to_pandas_dataframe()
predictions_0_1 = fitted_model.predict(X_validation)
class_probability = fitted_model.predict_proba(X_validation)
```

## `Explainability`


## * References 
[1] From  https://docs.microsoft.com/en-us/azure/machine-learning/how-to-machine-learning-interpretability-aml <br/>
[2] From https://docs.microsoft.com/en-us/python/api/azureml-widgets/azureml.widgets.rundetails?view=azure-ml-py <br/>
[3] From https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html <br/>
[4] From https://vitalflux.com/roc-curve-auc-python-false-positive-true-positive-rate/ <br/>
[5] From https://stackoverflow.com/questions/61184906/difference-between-predict-vs-predict-proba-in-scikit-learn <br/>
