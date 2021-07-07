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
import datetime
import time
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

# Show best run details 
RunDetails(best_run).show()
best_run_id = best_run.id 
experiment_name = 'Experiment_name'+date.today().strftime("%m%d_")
experiment_within_workspace= Experiment(ws, experiment_name)
best_run = Run(experiment_within_workspace, best_run_id)
print(best_run.get_metrics())

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
> `predict()` is used to predict the actual class (In your case one of 0, 1 or 1). <br/>
> `predict_proba()` is used to predict the class probabilities <br/>
> As an example: <br/>
> `predict()` would output class 0 since the class probability for 0 is 0.6. <br/>
> [0.6, 0.2, 0.2] is the output of `predict_proba` that simply denotes that the class probability  <br/>
> for classes 0, 1 and 2 are 0.6, 0.2 and 0.2 respectively. <br/>
> > -- <cite> Stack Overflow from [5] </cite>

``` python 
X_validation = validation_data.drop_columns(columns=target_column).to_pandas_dataframe()
y_validation = validation_data.keep_columns(columns=target_column, validate=True).to_pandas_dataframe()
predictions_0_1 = fitted_model.predict(X_validation)
class_probability = fitted_model.predict_proba(X_validation)
```

## `Explainability`

Based on [14], and [15]:

![model agnostic vs specific](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/model_agnostic_vs_specific.png)


![tabular_diagram_options](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/tabular_interpretation_techniques.png)


![tabular_table_explain_options](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/table_interpretability_technique_descrip_type.png)



Based on [6], [7], 

```python 
from azureml.interpret import ExplanationClient
import numpy as np
import pandas as pd

# Download the raw feature importances from the best run
client = ExplanationClient.from_run(best_run)
raw_explanations = client.download_model_explanation(raw=True)
print(raw_explanations.get_feature_importance_dict())
```

Based on [9], [10], [11], [12], and [13]:

- `mse` - Mean Squared Error:  tells you how close a regression line is to a set of points. It does this by taking the distances from the points to the regression line (these distances are the “errors”) and squaring them. The squaring is necessary to remove any negative signs. It also gives more weight to larger differences. It’s called the mean squared error as you’re finding the average of a set of errors. The lower the MSE, the better the forecast. 
- `mae` - Mean Absolute Error: measures the average magnitude of the errors in a set of predictions, without considering their direction. It's the average over the test sample of the absolute differences between prediction and actual observation where all individual differences have equal weight.
- `mape` - Mean Absolute Percentage Error: is a measure of how accurate a forecast system is. It measures this accuracy as a percentage, and can be calculated as the average absolute percent error for each time period minus actual values divided by actual values.
- `cosine` - Calculate similarity between dictionaries

```python
import sys
from tqdm import tqdm
import sklearn
from sklearn import preprocessing
from sklearn import metrics
from azureml.widgets import RunDetails
from azureml.core.run import Run
from azureml.core import Experiment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core import Workspace, Dataset

# Get best run files 
best_run_files = pd.DataFrame(data=best_run.get_file_names()) 
runs_ids = {}
run_metrics_details = {}

# Create dictionaries within experiments and key_metrics 
for run_n in tqdm(experiment_within_workspace.get_runs()):
    metrics = run_n.get_metrics()
    print(run_n)
    if 'metric_key_name' in metrics.keys():
        runs_ids[run_n.id] = run_n
        run_metrics_details[run_n.id] = metrics
        
# Register the model 
model_folder = './outputs/models'
model_name_selected= 'model_name'+date.today().strftime("%m%d_")
model = best_run.register_model(model_name=model_name_selected, model_path=model_folder+model_name_selected+'.pkl')
```

Based on [1], [13], and [16]:

```python 
from interpret.ext.blackbox import TabularExplainer
from interpret.ext.blackbox import MimicExplainer
# you can use one of the following four interpretable models as a global surrogate to the black box model
from interpret.ext.glassbox import LGBMExplainableModel
from interpret.ext.glassbox import LinearExplainableModel
from interpret.ext.glassbox import SGDExplainableModel
from interpret.ext.glassbox import DecisionTreeExplainableModel
from interpret.ext.blackbox import PFIExplainer
from azureml.interpret import ExplanationClient
from azureml.core.run import Run
from azureml.interpret import MimicWrapper
from azureml.train.automl.runtime.automl_explain_utilities import AutoMLExplainerSetupClass, automl_setup_model_explanations
from azureml.interpret.mimic_wrapper import MimicWrapper
from interpret_community.widget import ExplanationDashboard


# Global explanation 
ranked_global_values = raw_explanations.get_ranked_global_values()
ranked_global_names = raw_explanations.get_ranked_global_names()
print('Ranked Global Values: {}'.format(global_importance_values))
print('Ranked Global Names: {}'.format(global_importance_names))

# Local explanation
local_explanation = raw_explanations.explain_local(X_validation[0:5])
ranked_local__names = sorted(local_explanation.get_ranked_local_names())
ranked_local_values = sorted(local_explanation.get_ranked_local_values())

# Mimic Explainer
# "features" and "classes" fields are optional
# augment_data is optional and if true, oversamples the initialization examples to improve surrogate model accuracy to fit original model.  Useful for high-dimensional data where the number of rows is less than the number of columns.
# max_num_of_augmentations is optional and defines max number of times we can increase the input data size.
# LGBMExplainableModel can be replaced with LinearExplainableModel, SGDExplainableModel, or DecisionTreeExplainableModel
explainer = MimicExplainer(fitted_model, 
                           X_validations, 
                           LGBMExplainableModel, 
                           augment_data=True, 
                           max_num_of_augmentations=10, 
                           features=categorical_columns,
                           classes=target_column
                           )
                           
# Mimic Wrapper
explainer = MimicWrapper(ws, automl_explainer_setup_obj.automl_estimator,
                explainable_model=automl_explainer_setup_obj.surrogate_model,
                init_dataset=automl_explainer_setup_obj.X_transform, run=best_run,
                features=automl_explainer_setup_obj.engineered_feature_names,
                feature_maps=[automl_explainer_setup_obj.feature_map],
                classes=automl_explainer_setup_obj.classes,
                explainer_kwargs=automl_explainer_setup_obj.surrogate_model_params) 
#---- task_name:
# ************ regression
# ************ forecasting
# ************ classification
automl_explainer_setup_obj = automl_setup_model_explanations(fitted_model, X=X_train,
                                                             X_test=X_test, y=y_train,
                                                             task='task_name')                         
#---- Engineered Explanations
engineered_explanations = explainer.explain(['local', 'global'], eval_dataset=automl_explainer_setup_obj.X_test_transform)
print(engineered_explanations.get_feature_importance_dict()),
#---- Dashboard setup
ExplanationDashboard(engineered_explanations, automl_explainer_setup_obj.automl_estimator, datasetX=automl_explainer_setup_obj.X_test_transform)
#---- Raw Explanations
raw_explanations = explainer.explain(['local', 'global'], get_raw=True,
                                     raw_feature_names=automl_explainer_setup_obj.raw_feature_names,
                                     eval_dataset=automl_explainer_setup_obj.X_test_transform)
print(raw_explanations.get_feature_importance_dict()),
#---- Dashboard setup
ExplanationDashboard(raw_explanations, automl_explainer_setup_obj.automl_pipeline, datasetX=automl_explainer_setup_obj.X_test_raw)

  
# Tabular Explainer
run = best_run.get_context()
client = ExplanationClient.from_run(run)
# write code to get and split your data into train and test sets here
# write code to train your model here 
# explain predictions on your local machine
# "features" and "classes" fields are optional
explainer = TabularExplainer(fitted_model, 
                             X_validations.dropna(), 
                             features=categorical_columns, 
                             classes=target_column)
# explain overall model predictions (global explanation)
global_explanation = explainer.explain_global(X_validations.dropna())
# uploading global model explanation data for storage or visualization in webUX
# the explanation can then be downloaded on any compute
# multiple explanations can be uploaded
client.upload_model_explanation(global_explanation, comment='global explanation: all features')
# or you can only upload the explanation object with the top k feature info
#client.upload_model_explanation(global_explanation, top_k=2, comment='global explanation: Only top 2 features')
```

## * References 
[1] From  https://docs.microsoft.com/en-us/azure/machine-learning/how-to-machine-learning-interpretability-aml <br/>
[2] From https://docs.microsoft.com/en-us/python/api/azureml-widgets/azureml.widgets.rundetails?view=azure-ml-py <br/>
[3] From https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html <br/>
[4] From https://vitalflux.com/roc-curve-auc-python-false-positive-true-positive-rate/ <br/>
[5] From https://stackoverflow.com/questions/61184906/difference-between-predict-vs-predict-proba-in-scikit-learn <br/>
[6] From https://docs.microsoft.com/en-us/azure/machine-learning/how-to-machine-learning-interpretability-automl <br/>
[7] From https://github.com/MicrosoftDocs/azure-docs.es-es/blob/master/articles/machine-learning/how-to-machine-learning-interpretability-aml.md <br/>
[8] From https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html <br/>
[9] From https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/ <br/>
[10] From https://www.statisticshowto.com/probability-and-statistics/statistics-definitions/mean-squared-error/ <br/>
[11] From https://medium.com/human-in-a-machine-world/mae-and-rmse-which-metric-is-better-e60ac3bde13d#:~:text=Mean%20Absolute%20Error%20(MAE)%3A,individual%20differences%20have%20equal%20weight <br/>
[12] From https://www.statisticshowto.com/mean-absolute-percentage-error-mape/#:~:text=The%20mean%20absolute%20percentage%20error,values%20divided%20by%20actual%20values. <br/>
[13] From https://docs.microsoft.com/en-us/python/api/azureml-interpret/azureml.interpret.mimicwrapper?view=azure-ml-py <br/>
[14] From https://docs.microsoft.com/en-us/azure/machine-learning/how-to-machine-learning-interpretability <br/>
[15] From https://arxiv.org/pdf/2009.11698v1.pdf <br/>
[16] From https://github.com/MicrosoftDocs/azure-docs/blob/master/articles/machine-learning/how-to-machine-learning-interpretability-automl.md <br/>
