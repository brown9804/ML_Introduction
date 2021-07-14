# [Notes] Machine Learning Results/Explainability
----------

Costa Rica

Belinda Brown, belindabrownr04@gmail.com

Jan, 2021

----------

## `Display Results`

Considering that:
> In probability theory, an `event` is an `outcome or defined collection of outcomes of a random experiment`. 
> Since the collection of `all possible outcomes` to a random `experiment` is called the sample 
> `space`, another definiton of `event` is any `subset of a sample space`.
> > -- <cite> Statistics from [37] </cite>

```python 
from azureml.widgets import RunDetails
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, au
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_recall_fscore_support
import lime
import lime.lime_tabular
import lightgbm as lgb
import pandas as pd
```

### `→ Display Details:`
Based on [2], and [3]:
``` python 
# Show best run details 
RunDetails(best_run).show()
best_run_id = best_run.id 
experiment_name = 'Experiment_name'+date.today().strftime("_%m%d")
experiment_within_workspace= Experiment(ws, experiment_name)
best_run = Run(experiment_within_workspace, best_run_id)
print(best_run.get_metrics())
```


### `→ Confusion Matrix:`

Based on [20]:
![confusion matrix explain](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/confusion_matrix_explain.png)

``` python
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
From [38]:

![confusion_matrix_good_bad](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/confusion_matrix_good_bad.png)


### `→ ROC/AUC:`

From [21]:

![roc uac explain](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/roc_auc_explain.png)

Based on [4]:
```python 
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

From [38]:

![roc_good_bad](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/roc_good_bad.png)

### `→ Lift Curve:`
 
> Is a way of visualizing the performance of a classification model.
> `How does my model compare to random guessing given X number of samples`
> > -- <cite> Perkowski from [36] </cite>

From [38]:

![lift_good_bad](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/lift_good_bad.png)

### `→ Precision-recall curve:`

> The precision-recall curve plots the `relationship between precision and recall` as 
> the `decision threshold changes`. Recall is the ability of a model to detect all positive 
> samples and precision is the ability of a model to avoid labeling negative samples as 
> positive. Some business problems might require higher recall and some higher precision 
> depending on the relative importance of avoiding false negatives vs false positives.
> > -- <cite> MicrosoftDocs from [38] </cite>

From [38]:

![precisionrecall_good_bad](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/precisionrecall_good_bad.png)


### `→ Calibration curve:`

`Proportion of confidence by confidence level`. `Calibrated - the level has the expected value`

> The calibration curve plots a `model's confidence` in its `predictions against` the proportion 
> of `positive samples at each confidence level`. A `well-calibrated` model will correctly classify
> 100% of the predictions to which it assigns 100% confidence, 50% of the predictions it assigns 
> 50% confidence, 20% of the predictions it assigns a 20% confidence, and so on. A perfectly 
> calibrated model will have a calibration curve following the y = x line where the model perfectly 
> predicts the probability that samples belong to each class. <br/>
> An `over-confident model` will over-predict probabilities close to zero and one,
> rarely being uncertain about the class of each sample and the `calibration curve` will look similar 
> to `backward "S"`. An `under-confident` model will assign a lower probability on average to the class
> it predicts and the associated calibration curve will look `similar to an "S"`. The calibration 
> curve does not depict a model's ability to classify correctly, but instead its ability to
> correctly assign confidence to its predictions. A bad model can still have a good calibration
> curve if the model correctly assigns low confidence and high uncertainty.
> > -- <cite> MicrosoftDocs from [1] </cite>


From [38]:

![calibration_curve_good_bad](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/calibration_curve_good_bad.png)

### `→ Cumulative Gains:`
> Shows the total number of events captured by a model over a given number of samples.
> `how many events can I expect given X number of samples`
> > -- <cite> Perkowski from [36] </cite>

From [38]:

![comulative_gains_good_bad](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/comulative_gains_good_bad.png)

### `→ Zscore:`

From [42]:

![zscore](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/zscore.png)

From [45]:

![mean_std_desv_formulas](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/mean_std_desv_formulas.png)


From [43]:

![zscore_graph](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/zscore_graph.png)

From [44]:

![example_using_zscore](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/example_using_zscore.png)

From [44]:

```python 
# Prepare Data
df = pd.read_csv("https://github.com/selva86/datasets/raw/master/mtcars.csv")
x = df.loc[:, ['mpg']]
df['mpg_z'] = (x - x.mean())/x.std() # ZSCORE formula
df['colors'] = ['red' if x < 0 else 'green' for x in df['mpg_z']]
df.sort_values('mpg_z', inplace=True)
df.reset_index(inplace=True)

# Draw plot
plt.figure(figsize=(14,10), dpi= 80)
plt.hlines(y=df.index, xmin=0, xmax=df.mpg_z, color=df.colors, alpha=0.4, linewidth=5)

# Decorations
plt.gca().set(ylabel='$Model$', xlabel='$Mileage$')
plt.yticks(df.index, df.cars, fontsize=12)
plt.title('Diverging Bars of Car Mileage', fontdict={'size':20})
plt.grid(linestyle='--', alpha=0.5)
plt.show()
```

### `→ Other Metrics:`

From [22]:

![Other metrics explain](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/other_metrics_explain.png)

Difference between accuracy and precision, from [23]:

![diff_accuracy_precision](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/accuracy_vs_precision_explain.png)

Considering:
> `predict()` is used to predict the actual class (example of clases 0, 1 or 2). <br/>
> `predict_proba()` is used to predict the class probabilities <br/>
> As an example: <br/>
> `predict()` would output class 0 since the class probability for 0 is 0.6. <br/>
> [0.6, 0.2, 0.2] is the output of `predict_proba` that simply denotes that the class probability  <br/>
> for classes 0, 1 and 2 are 0.6, 0.2 and 0.2 respectively. <br/>
> > -- <cite> Based on Stack Overflow from [5] </cite>

Based on [18], [19]:
``` python 
# For precision recall average can be:
# - weighted
# - macro
# - micro 
X_validation = validation_data.drop_columns(columns=target_column).to_pandas_dataframe()
y_validation = validation_data.keep_columns(columns=target_column, validate=True).to_pandas_dataframe()
predictions_0_1 = fitted_model.predict(X_validation) # calculate y predictions 
class_probability = fitted_model.predict_proba(X_validation) # calculate probability by classes 
precision_recall_fscore = precision_recall_fscore_support(y_validation, predictions_0_1, average='weighted')
accuracy = accuracy_score(y_validation, predictions_0_1, normalize=False) # it's set by defult (true) so expected value is 0-1
```

## `Explainability`

From [40]:

![explainability_definiton](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/explainability_definiton.png)


From [33]:

![how_to_explain_your_model](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/how_to_explain_your_model.png)

From [33]:

![direct_vs_meta_explainers](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/direct_vs_meta_explainers.png)

From [33]:

![ml_interpretability](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/ml_interpretability_table_general.png)

Based on [14], and [15]:

![model agnostic vs specific](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/model_agnostic_vs_specific.png)


 ```python 
import numpy as np
import pandas as pd
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
from azureml.interpret import ExplanationClient
 ```

### `→ Download Explanation:`

Based on [6], [7]:

```python 
# Download the raw feature importances from the best run
client = ExplanationClient.from_run(best_run)
raw_explanations = client.download_model_explanation(raw=True)
print(raw_explanations.get_feature_importance_dict())
```

### `→ Dictionary within experiment keymetrics:`


Metric rules for point forecasting, from [24]:

![mape, mse, mae, r2, explain](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/Three-metric-rules-for-point-forecasting_mape_mae_mse_cosine.png)


Based on [9], [10], [11], [12], and [13]:

- `mse` - Mean Squared Error:  tells you how close a regression line is to a set of points. It does this by taking the distances from the points to the regression line (these distances are the “errors”) and squaring them. The squaring is necessary to remove any negative signs. It also gives more weight to larger differences. It’s called the mean squared error as you’re finding the average of a set of errors. The lower the MSE, the better the forecast. 
- `mae` - Mean Absolute Error: measures the average magnitude of the errors in a set of predictions, without considering their direction. It's the average over the test sample of the absolute differences between prediction and actual observation where all individual differences have equal weight.
- `mape` - Mean Absolute Percentage Error: is a measure of how accurate a forecast system is. It measures this accuracy as a percentage, and can be calculated as the average absolute percent error for each time period minus actual values divided by actual values.
- `cosine` - Calculate similarity between dictionaries

```python
# Print get run content
run = best_run.get_context()
print(str(run))
# Get best run files 
best_run_files = pd.DataFrame(data=best_run.get_file_names()) 
runs_ids = {}
run_metrics_details = {}

# Filtering by keymetric
for run_n in tqdm(experiment_within_workspace.get_runs()):
    metrics = run_n.get_metrics()
    print(run_n)
    if 'metric_key_name' in metrics.keys():
        runs_ids[run_n.id] = run_n
        run_metrics_details[run_n.id] = metrics
```

### `→ Global/Local explanation:`

Based on [26]:

![local_vs_global_explain](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/local_vs_global_explain.png)

Based on [1], [13], and [16]:

Explainability Model Type Output:
- `tabular_explainer` (Global explanation MicrosoftDocs recommended)
- `mimic_explainer` (Global/Local)
- `mimic_wrapper_explainer` (Global/Local)
- `pfi_explainer` (Global/ no Local support)

From [33]:

![global_explain_output_example](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/global_explain_output_example.png)

From [33]:

![local_explain_output_example](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/local_explain_output_example.png)


## Meta Explainers

### `→ Tabular Explainer:`

 >  Tabular Explainer is an explainer that acts as a wrapper around various SHAP explainer algorithms, automatically choosing the one that is most appropriate for your model architecture.
> > -- <cite> CourseHero from [32] </cite>


> TabularExplainer `calls one of the three SHAP 
> explainers underneath` (TreeExplainer, DeepExplainer, or KernelExplainer). <br/>
> TabularExplainer `automatically selects the most appropriate` one for 
>  your use case, but you can call each of its three underlying 
>  explainers directly.
> > -- <cite> MicrosoftDocs from [1] </cite>

 Based on [1], [13], and [16]:
 
 Task names:
- regression
- forecasting
- classification

Based on [14], and [15]:

![tabular_diagram_options](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/tabular_interpretation_techniques.png)
 
 From [1]:
 
 ```python 
# load breast cancer dataset, a well-known small dataset that comes with scikit-learn
from sklearn.datasets import load_breast_cancer
from sklearn import svm
from sklearn.model_selection import train_test_split
from azureml.interpret import ExplanationClient
from azureml.core.run import Run
from interpret.ext.blackbox import TabularExplainer

breast_cancer_data = load_breast_cancer()
classes = breast_cancer_data.target_names.tolist()

# split data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(breast_cancer_data.data,            
                                                    breast_cancer_data.target,  
                                                    test_size=0.2,
                                                    random_state=0)
clf = svm.SVC(gamma=0.001, C=100., probability=True)
fitted_model = clf.fit(x_train, y_train)

# automl_explainer_setup_obj = automl_setup_model_explanations(fitted_model, X=X_train, 
#                                                              X_test=X_test, y=y_train, 
#                                                              task='classification')

# bes_run_context = best_run.get_context()
# client = ExplanationClient.from_run(bes_run_context)

# write code to get and split your data into train and test sets here
# write code to train your model here 
    
tabular_explainer = TabularExplainer(fitted_model, 
                             x_train.dropna().iloc[0:5000,:], 
                             features=automl_explainer_setup_obj.engineered_feature_names,
                             classes=automl_explainer_setup_obj.classes )

# explain overall model predictions (global explanation)
global_explanation = tabular_explainer.explain_global(x_train.dropna().iloc[0:5000,:])
# uploading global model explanation data for storage or visualization in webUX
# the explanation can then be downloaded on any compute
# multiple explanations can be uploaded
client.upload_model_explanation(global_explanation, comment='global explanation: all features')
# or you can only upload the explanation object with the top k feature info
#client.upload_model_explanation(global_explanation, top_k=2, comment='global explanation: Only top 2 features')
```

### `→ Mimic Explainer:`

From [14]:

![mimic_table_explain_options](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/table_interpretability_technique_description_type_mimic_explainer.png)


From [1]:

```python 
# load breast cancer dataset, a well-known small dataset that comes with scikit-learn
from sklearn.datasets import load_breast_cancer
from sklearn import svm
from sklearn.model_selection import train_test_split
from interpret.ext.blackbox import MimicExplainer
# you can use one of the following four interpretable models as a global surrogate to the black box model
from interpret.ext.glassbox import LGBMExplainableModel
from interpret.ext.glassbox import LinearExplainableModel
from interpret.ext.glassbox import SGDExplainableModel
from interpret.ext.glassbox import DecisionTreeExplainableModel

breast_cancer_data = load_breast_cancer()
classes = breast_cancer_data.target_names.tolist()

# split data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(breast_cancer_data.data,            
                                                    breast_cancer_data.target,  
                                                    test_size=0.2,
                                                    random_state=0)
clf = svm.SVC(gamma=0.001, C=100., probability=True)
model = clf.fit(x_train, y_train)

# "features" and "classes" fields are optional
# augment_data is optional and if true, oversamples the initialization examples to improve surrogate model accuracy to fit original model.  Useful for high-dimensional data where the number of rows is less than the number of columns.
# max_num_of_augmentations is optional and defines max number of times we can increase the input data size.
# LGBMExplainableModel can be replaced with LinearExplainableModel, SGDExplainableModel, or DecisionTreeExplainableModel
mimic_explainer = MimicExplainer(model, 
                           x_train, 
                           LGBMExplainableModel, 
                           augment_data=True, 
                           max_num_of_augmentations=10, 
                           features=breast_cancer_data.feature_names, 
                           classes=classes)
# Local explanation                           
# get explanation for the first data point in the test set
local_explanation = explainer.explain_local(x_test[0:5])
# sorted feature importance values and feature names
sorted_local_importance_names = local_explanation.get_ranked_local_names()
sorted_local_importance_values = local_explanation.get_ranked_local_values()   

# Global explanation
# you can use the training data or the test data here, but test data would allow you to use Explanation Exploration
global_explanation = explainer.explain_global(x_test)
# sorted feature importance values and feature names
sorted_global_importance_values = global_explanation.get_ranked_global_values()
sorted_global_importance_names = global_explanation.get_ranked_global_names()
dict(zip(sorted_global_importance_names, sorted_global_importance_values))
# alternatively, you can print out a dictionary that holds the top K feature names and values -> need to print in a text file since it's a lot of information
# global_explanation.get_feature_importance_dict()
```

### `→ Mimic Wrapper:`

Based on `best_run`. 

A wrapper explainer is which reduces the number of function calls necessary to use the explain model package.

Based on [1], [13], and [16]:

```python 
#---- task_name:
# ************ regression
# ************ forecasting
# ************ classification

best_run, fitted_model = remote_run.get_output()

from interpret.ext.glassbox import LGBMExplainableModel
from azureml.interpret.mimic_wrapper import MimicWrapper
from azureml.train.automl.runtime.automl_explain_utilities import AutoMLExplainerSetupClass, automl_setup_model_explanations
from interpret_community.widget import ExplanationDashboard

automl_explainer_setup_obj = automl_setup_model_explanations(fitted_model, X=X_train,
                                                             X_test=X_test, y=y_train,
                                                             task='task_name')

# LGBMExplainableModel can be replaced with LinearExplainableModel, SGDExplainableModel, or DecisionTreeExplainableModel
mimic_wrapper_explainer = MimicWrapper(ws, automl_explainer_setup_obj.automl_estimator, LGBMExplainableModel,
                         init_dataset=automl_explainer_setup_obj.X_transform, run=best_run,
                         features=automl_explainer_setup_obj.engineered_feature_names,
                         feature_maps=[automl_explainer_setup_obj.feature_map],
                         classes=automl_explainer_setup_obj.classes)
# Local/Global Explanation                       
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
```

## Direct Explainers 

`SHAP`: SHapley Additive exPlanations 

> SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any
> machine learning model. It connects optimal credit allocation with local explanations using 
> the classic Shapley values from game theory and their related extensions 
> (see papers for details and citations).
> > -- <cite> SHAP from [39] </cite>

From [39]: 
![shap](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/shap.png)

From [14]:

![tabular_table_explain_options](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/table_interpretability_technique_descrip_type_tabular.png)

### `→ SHAP Tree Explainer:`

From [31]:

```python 
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import shap
# print the JS visualization code to the notebook
shap.initjs()
#------> Load data set 
X,y = shap.datasets.adult()
X_display,y_display = shap.datasets.adult(display=True)
# create a train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
d_train = lgb.Dataset(X_train, label=y_train)
d_test = lgb.Dataset(X_test, label=y_test)
#------> Train the model 
params = {
    "max_bin": 512,
    "learning_rate": 0.05,
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 10,
    "verbose": -1,
    "min_data": 100,
    "boost_from_average": True
}
model = lgb.train(params, d_train, 10000, valid_sets=[d_test], early_stopping_rounds=50, verbose_eval=1000)
#------> Explain predictions
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
#------> Visualize a single prediction
shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X_display.iloc[0,:])
#------> Visualize many predictions
shap.force_plot(explainer.expected_value[1], shap_values[1][:1000,:], X_display.iloc[:1000,:])
#------> SHAP Summary Plot
shap.summary_plot(shap_values, X)
#------> SHAP Dependence Plots
for name in X_train.columns:
    shap.dependence_plot(name, shap_values[1], X, display_features=X_display)
```

### `→ SHAP Deep Explainer:`
Based on [30], [41]:

```python 
!pip install tensorflow==1.2.0 --ignore-installed

!pip uninstall tensorflow -y
!echo "y" | conda create --name tensorflow-env python=3.8 pip
!conda activate tensorflow-env
!pip install tensorflow

import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf    
tf.compat.v1.disable_v2_behavior() # <-- HERE !
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential
from tensorflow.keras import optimizers

print("SHAP version is:", shap.__version__)
print("Tensorflow version is:", tf.__version__)
X_train, X_test, Y_train, Y_test = train_test_split(
    *shap.datasets.iris(), test_size=0.2, random_state=0
)
Y_train = to_categorical(Y_train, num_classes=3)
Y_test = to_categorical(Y_test, num_classes=3)
# Define baseline model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(8, input_dim=len(X_train.columns), activation="relu"))
model.add(tf.keras.layers.Dense(3, activation="softmax"))
# model.summary()
# compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
hist = model.fit(X_train, Y_train, batch_size=5, epochs=200, verbose=0)
# select a set of background examples to take an expectation over
background = X_train.iloc[np.random.choice(X_train.shape[0], 100, replace=False)]
explainer = shap.DeepExplainer(
    (model.layers[0].input, model.layers[-1].output), background
)
shap_values = explainer.shap_values(X_test[:3].values) # <-- HERE !
# print the JS visualization code to the notebook
shap.initjs()
shap.force_plot(
    explainer.expected_value[0], shap_values[0][0], feature_names=X_train.columns
)
```

### `→ SHAP Linear Explainer:`

From [34]:

```python
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import shap
shap.initjs()
#------> Load the IMDB dataset
corpus,y = shap.datasets.imdb()
corpus_train, corpus_test, y_train, y_test = train_test_split(corpus, y, test_size=0.2, random_state=7)
vectorizer = TfidfVectorizer(min_df=10)
X_train = vectorizer.fit_transform(corpus_train)
X_test = vectorizer.transform(corpus_test)
#------> Fit a linear logistic regression model
model = sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1)
model.fit(X_train, y_train)
#------> Explain the linear model
explainer = shap.LinearExplainer(model, X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_test)
X_test_array = X_test.toarray() # we need to pass a dense version for the plotting functions
#------> Summarize the effect of all the features
shap.summary_plot(shap_values, X_test_array, feature_names=vectorizer.get_feature_names())
#------> Explain the first review's sentiment prediction
ind = 0
shap.force_plot(
    explainer.expected_value, shap_values[ind,:], X_test_array[ind,:],
    feature_names=vectorizer.get_feature_names()
)
print("Positive" if y_test[ind] else "Negative", "Review:")
print(corpus_test[ind])
#------> Explain the second review's sentiment prediction
ind = 1
shap.force_plot(
    explainer.expected_value, shap_values[ind,:], X_test_array[ind,:],
    feature_names=vectorizer.get_feature_names()
)
print("Positive" if y_test[ind] else "Negative", "Review:")
print(corpus_test[ind])
#------> Explain the third review's sentiment prediction
ind = 2
shap.force_plot(
    explainer.expected_value, shap_values[ind,:], X_test_array[ind,:],
    feature_names=vectorizer.get_feature_names()
)
print("Positive" if y_test[ind] else "Negative", "Review:")
print(corpus_test[ind])
```

### `→ SHAP Kernel Explainer:`

From [35]:

```python
# Iris classification with scikit-learn
#------> Load the data
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import shap
import time
X_train,X_test,Y_train,Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
# rather than use the whole training set to estimate expected values, we could summarize with
# a set of weighted kmeans, each weighted by the number of points they represent. But this dataset
# is so small we don't worry about it
#X_train_summary = shap.kmeans(X_train, 50)
def print_accuracy(f):
    print("Accuracy = {0}%".format(100*np.sum(f(X_test) == Y_test)/len(Y_test)))
    time.sleep(0.5) # to let the print get out before any progress bars
shap.initjs()
#------> K-nearest neighbors
knn = sklearn.neighbors.KNeighborsClassifier()
knn.fit(X_train, Y_train)
print_accuracy(knn.predict)
# --> Explain a single prediction from the test set
explainer = shap.KernelExplainer(knn.predict_proba, X_train)
shap_values = explainer.shap_values(X_test.iloc[0,:])
shap.force_plot(explainer.expected_value[0], shap_values[0], X_test.iloc[0,:])
# --> Explain all the predictions in the test set
shap_values = explainer.shap_values(X_test)
shap.force_plot(explainer.expected_value[0], shap_values[0], X_test)
#------> Support vector machine with a linear kernel
svc_linear = sklearn.svm.SVC(kernel='linear', probability=True)
svc_linear.fit(X_train, Y_train)
print_accuracy(svc_linear.predict)
# explain all the predictions in the test set
explainer = shap.KernelExplainer(svc_linear.predict_proba, X_train)
shap_values = explainer.shap_values(X_test)
shap.force_plot(explainer.expected_value[0], shap_values[0], X_test)
#------> Support vector machine with a radial basis function kernel
svc_linear = sklearn.svm.SVC(kernel='rbf', probability=True)
svc_linear.fit(X_train, Y_train)
print_accuracy(svc_linear.predict)
# explain all the predictions in the test set
explainer = shap.KernelExplainer(svc_linear.predict_proba, X_train)
shap_values = explainer.shap_values(X_test)
shap.force_plot(explainer.expected_value[0], shap_values[0], X_test)
#------> Logistic regression
linear_lr = sklearn.linear_model.LogisticRegression()
linear_lr.fit(X_train, Y_train)
print_accuracy(linear_lr.predict)
# explain all the predictions in the test set
explainer = shap.KernelExplainer(linear_lr.predict_proba, X_train)
shap_values = explainer.shap_values(X_test)
shap.force_plot(explainer.expected_value[0], shap_values[0], X_test)
#------> Decision tree
import sklearn.tree 
dtree = sklearn.tree.DecisionTreeClassifier(min_samples_split=2)
dtree.fit(X_train, Y_train)
print_accuracy(dtree.predict)
# explain all the predictions in the test set
explainer = shap.KernelExplainer(dtree.predict_proba, X_train)
shap_values = explainer.shap_values(X_test)
shap.force_plot(explainer.expected_value[0], shap_values[0], X_test)
#------> Random forest
from sklearn.ensemble import RandomForestClassifier
rforest = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
rforest.fit(X_train, Y_train)
print_accuracy(rforest.predict)
# explain all the predictions in the test set
explainer = shap.KernelExplainer(rforest.predict_proba, X_train)
shap_values = explainer.shap_values(X_test)
shap.force_plot(explainer.expected_value[0], shap_values[0], X_test)
#------> Neural network
from sklearn.neural_network import MLPClassifier
nn = MLPClassifier(solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(5, 2), random_state=0)
nn.fit(X_train, Y_train)
print_accuracy(nn.predict)
# explain all the predictions in the test set
explainer = shap.KernelExplainer(nn.predict_proba, X_train)
shap_values = explainer.shap_values(X_test)
shap.force_plot(explainer.expected_value[0], shap_values[0], X_test)
```

### `→ PFI Explainer:`

From [14]:

![permutation_table_explain_options](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/table_interpretability_technique_description_type_permutation_explainer.png)

From on [1];

```python 
# load breast cancer dataset, a well-known small dataset that comes with scikit-learn
from sklearn.datasets import load_breast_cancer
from sklearn import svm
from sklearn.model_selection import train_test_split
from interpret.ext.blackbox import PFIExplainer

breast_cancer_data = load_breast_cancer()
classes = breast_cancer_data.target_names.tolist()
# split data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(breast_cancer_data.data,            
                                                    breast_cancer_data.target,  
                                                    test_size=0.2,
                                                    random_state=0)
clf = svm.SVC(gamma=0.001, C=100., probability=True)
model = clf.fit(x_train, y_train)

# "features" and "classes" fields are optional
pfi_explainer = PFIExplainer(model,
                         features=breast_cancer_data.feature_names, 
                         classes=classes)

# if you used the PFIExplainer in the previous step, use the next line of code instead
global_explanation = pfi_explainer.explain_global(x_train, true_labels=y_train)

# sorted feature importance values and feature names
sorted_global_importance_values = global_explanation.get_ranked_global_values()
sorted_global_importance_names = global_explanation.get_ranked_global_names()
dict(zip(sorted_global_importance_names, sorted_global_importance_values))
# alternatively, you can print out a dictionary that holds the top K feature names and values -> need to print in a text file since it's a lot of information
# global_explanation.get_feature_importance_dict()                     
```

### `→ LIME Explainer:`

> LIME ( Local Interpretable `Model-agnostic Explanations` )
> is a novel explanation technique that explains the prediction of any classifier 
> in an interpretable and faithful manner by learning an 
> interpretable model locally around the prediction.
> What has LIME had to offer on model interpretability?
> 1. A consistent model agnostic explainer [ LIME ].
> 2. A method to select a representative set with explanations [ SP-LIME ] 
> to make sure the model behaves consistently while
> replicating human logic. `This representative set would provide an 
> intuitive global understanding of the model`.
> > -- <cite> Towards DataScience from [29] </cite>


Example from [29]:

```python 
import lime
import lime.lime_tabular
import pandas as pd
import numpy as np
import lightgbm as lgb
# For converting textual categories to integer labels 
from sklearn.preprocessing import LabelEncoder
# for creating train test split
from sklearn.model_selection import train_test_split

# specify your configurations as a dict
lgb_params = {
    'task': 'train',
    'boosting_type': 'goss',
    'objective': 'binary',
    'metric':'binary_logloss',
    'metric': {'l2', 'auc'},
    'num_leaves': 50,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'verbose': None,
    'num_iteration':100,
    'num_threads':7,
    'max_depth':12,
    'min_data_in_leaf':100,
    'alpha':0.5}

# reading the titanic data
df_titanic = pd.read_csv(r'/Users/300011432/Downloads/all/train.csv')

# data preparation
df_titanic.fillna(0,inplace=True)
le = LabelEncoder()
feat = ['PassengerId', 'Pclass_le', 'Sex_le','SibSp_le', 'Parch','Fare']

# label encoding textual data
df_titanic['Pclass_le'] = le.fit_transform(df_titanic['Pclass'])
df_titanic['SibSp_le'] = le.fit_transform(df_titanic['SibSp'])
df_titanic['Sex_le'] = le.fit_transform(df_titanic['Sex'])

# using train test split to create validation set
X_train,X_test,y_train,y_test = train_test_split(df_titanic[feat],df_titanic[['Survived']],test_size=0.3)

# def lgb_model(X_train,y_train,X_test,y_test,lgb_params):
# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test)

# training the lightgbm model
model = lgb.train(lgb_params,lgb_train,num_boost_round=20,valid_sets=lgb_eval,early_stopping_rounds=5)

# this is required as LIME requires class probabilities in case of classification example
# LightGBM directly returns probability for class 1 by default 
def prob(data):
    return np.array(list(zip(1-model.predict(data),model.predict(data))))
    
explainer = lime.lime_tabular.LimeTabularExplainer(df_titanic[model.feature_name()].astype(int).values,  
mode='classification',training_labels=df_titanic['Survived'],feature_names=model.feature_name())

# asking for explanation for LIME model
i = 1
exp = explainer.explain_instance(df_titanic.loc[i,feat].astype(int).values, prob, num_features=5)
```

### `→ Model registration:`

![pkl visual explain](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/pkl_explain.png)

```python
model_folder = './outputs/models'
model_name_selected= 'model_name'+date.today().strftime("__%m%d")
model = best_run.register_model(model_name=model_name_selected, model_path=model_folder+model_name_selected+'.pkl')
```

## * References 
[1] From https://docs.microsoft.com/en-us/azure/machine-learning/how-to-machine-learning-interpretability-aml <br/>
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
[17] From https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/automated-machine-learning/classification-credit-card-fraud/auto-ml-classification-credit-card-fraud.ipynb <br/>
[18] From https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html <br/>
[19] From https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html <br/>
[20] From https://ailearnerhub.com/2020/05/10/what-is-the-confusion-matrix/ <br/>
[21] From https://www.youtube.com/watch?v=afQ_DyKMxUo <br/>
[22] From https://ai-ml-analytics.com/classification-metrics-in-machine-learning/ <br/>
[23] From https://sketchplanations.com/accuracy-and-precision <br/>
[24] From https://www.researchgate.net/figure/Three-metric-rules-for-point-forecasting_tbl1_314201097 <br/>
[25] From https://atrium.ai/resources/build-and-deploy-a-docker-containerized-python-machine-learning-model-on-heroku/ <br/>
[26] From https://spectra.pub/ml/demystify-post-hoc-explainability <br/>
[27] From https://scikit-learn.org/stable/modules/permutation_importance.html <br/>
[28] From https://github.com/interpretml/interpret-community <br/>
[29] From https://towardsdatascience.com/decrypting-your-machine-learning-model-using-lime-5adc035109b5 <br/>
[30] From https://github.com/slundberg/shap <br/>
[31] From https://slundberg.github.io/shap/notebooks/tree_explainer/Census%20income%20classification%20with%20LightGBM.html#SHAP-Summary-Plot <br/>
[32] From https://www.coursehero.com/file/p53lo034/Visualizing-Summary-Importance-Switching-to-the-Summary-Importance/ <br/>
[33] From https://www.slideshare.net/FrancescaLazzeriPhD/using-auto-ml-to-automate-selection-of-machine-learning-models-and-hyperparameters-ai-conference-sj-2019 <br/>
[34] From https://slundberg.github.io/shap/notebooks/linear_explainer/Sentiment%20Analysis%20with%20Logistic%20Regression.html <br/>
[35] From https://slundberg.github.io/shap/notebooks/Iris%20classification%20with%20scikit-learn.html <br/>
[36] From https://www.quora.com/Whats-Lift-curve <br/>
[37] From https://www.statistics.com/glossary/event/ <br/>
[38] From https://docs.microsoft.com/en-us/azure/machine-learning/how-to-understand-automated-ml <br/>
[39] From https://shap.readthedocs.io/en/latest/index.html <br/>
[40] From https://human-centered.ai/wordpress/wp-content/uploads/2020/02/04-Explainable-AI-Global-Local-Antehoc-Posthoc-Overview.pdf <br/>
[41] From https://stackoverflow.com/questions/66814523/shap-deepexplainer-with-tensorflow-2-4-error <br/>
[42] From https://slideplayer.com/slide/1388845/ <br/>
[43] From http://www.robertbarrington.net/wp-content/uploads/2013/04/ZScores.jpg <br/>
[44] From https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/ <br/>
[45] From https://www.kristakingmath.com/blog/mean-variance-standard-deviation <br/>
