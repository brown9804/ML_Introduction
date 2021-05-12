# [Notes] Machine Learning 

----------

Costa Rica

Belinda Brown, belindabrownr04@gmail.com

Jan, 2021

----------

It is a branch of artificial intelligence that allows machines to learn without being expressly programmed to do so. The idea is to identify the patterns between the data to make predictions. That is, the relationships between the columns are analyzed.

## Dictionary
`Target Variable`: The target variable of a dataset is the feature of a dataset about which you want to gain a deeper understanding. [8]

## 3 main Types of Learning:

|   Type  | Category | Definition |
|---|---|---|
| Supervised | Classification and Regression | This algorithms learns the input pattern and generates the expected output. We have expected output associated with our input data. [9] |
| Unsupervised | Clustering and Association | The task of machine is to group unsorted information according to similarities, patterns and differences without any prior training of data. [9]|
| Reinforcement | *  | Reinforcement learning works by putting the algorithm in a work environment with an interpreter and a reward system. In every iteration of the algorithm, the output result is given to the interpreter, which decides whether the outcome is favorable or not. [12] |

From [18]:
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/types_of_ml_based_in_objective.png)

From [14]
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/ml_types_input_output.png)

From [11]
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/machine_learning_types.png)

### Type #1 Supervised Learning 

|   Use  | % | 
|---|---|
| Training | 80 | 
| Validation | 20 |

From [14]:
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/supervised_model_in_out.png)

From [19]:
<br/>

|   Category  | Definition | 
|---|---|
| Classification | Supervised learning problem that involves predicting a class label. [19]| 
| Regression | Supervised learning problem that involves predicting a numerical label. [19]|


From [15]:
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/regression_vs_classification_supervised_learning.png)


From [22]:
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/supervised_general_flow.png)

#### 1.1 Decision Trees
> A decision tree is a supervised learning method for classification. 
> Algorithms of this variety create trees that predict the result of an 
> input vector based on decision rules inferred from the features present 
> in the data. Decision trees are useful because they’re easy to visualize 
> so you can understand the factors that lead to a result.
>
> Two types of models exist for decision trees: <br/>
> Classification trees, where the target variable is a discrete value and the leaves represent class labels (as shown in the example tree), and <br/>
> Regression trees, where the target variable can take continuous values.
> 
> > -- <cite> From [14] IBM </cite>
> > 

From [14]:
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/typical_decision_tree.png)

##### 1.1.1 Classification Trees (Discrete/Binary - Yes/No types)

> A categorical variable decision tree includes categorical target 
> variables that are divided into categories. For example, the categories 
> can be yes or no. The categories mean that every stage of the decision 
> process falls into one of the categories, and there are no in-betweens.
> 
> > -- <cite> From [13] CFI </cite>


##### 1.1.1 Regression Trees (Continuous data types)

> A continuous variable decision tree is a decision tree with a continuous 
> target variable. For example, the income of an individual whose income is
> unknown can be predicted based on available information such as their 
> occupation, age, and other continuous variables.
> 
> > -- <cite> From [13] CFI </cite>

#### 1.2 Logictic Regression
From [21]:
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/linear_logistic_regression_graph.png)

Assumptions:

> `Binary Output Variable`: This might be obvious as we have already mentioned it, 
> but logistic regression is intended for binary (two-class) classification problems. 
> It will predict the probability of an instance belonging to the default class,
> which can be snapped into a 0 or 1 classification.
> 
> `Remove Noise`: Logistic regression assumes no error in the output variable (y), 
> consider removing outliers and possibly misclassified instances from your training data.
> 
> `Gaussian Distribution`: Logistic regression is a linear algorithm (with a non-linear transform 
> on output). It does assume a linear relationship between the input variables with the output.
> Data transforms of your input variables that better expose this linear relationship can result
> in a more accurate model. For example, you can use log, root, Box-Cox and other univariate 
> transforms to better expose this relationship.
> 
> `Remove Correlated Input`: Like linear regression, the model can overfit if you have multiple 
> highly-correlated inputs. Consider calculating the pairwise correlations between all inputs 
> and removing highly correlated inputs.
> 
> `Fail to Converge`: It is possible for the expected likelihood estimation process that learns 
> the coefficients to fail to converge. This can happen if there are many highly correlated inputs
> in your data or the data is very sparse (e.g. lots of zeros in your input data).

> > -- <cite> From [20] Machine Learning Mastery </cite>


#### 1.3 Linear Regression

From [21]:
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/supervised_logistic_vs_linear_regression.png)

### Type #2 Unsupervised Learning 

From [14]:
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/unsupervised_logic_data_in_out.png)


From [16]:
<br/>

|   Categegory  | Definition | 
|---|---|
| Clustering | Clustering is a method of grouping the objects into clusters such that objects with most similarities remains into a group and has less or no similarities with the objects of another group. Cluster analysis finds the commonalities between the data objects and categorizes them as per the presence and absence of those commonalities. [16]| 
| Association | An association rule is an unsupervised learning method which is used for finding the relationships between variables in the large database. It determines the set of items that occurs together in the dataset. Association rule makes marketing strategy more effective. Such as people who buy X item (suppose a bread) are also tend to purchase Y (Butter/Jam) item. [16]|

From [22]:
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/unsupervised_general_flow.png)

### Type #3 Reinforcement Learning 
From [14]:
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/reinforcement_logic_data_in_out.png)


From [17]:
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/reinforcement_flow_basic_1.png)


From [17]:
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/reinforcement_flow_basic_2.png)


From [22]:
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/reinforcement_general_flow.png)

## How to choose a Performance Metric


From [10]:
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/How-to-Choose-a-Metric-for-Imbalanced-Classification-latest.png)

### Evaluate Logistic regression model
> 1. `AIC (Akaike Information Criteria)` – The analogous metric of adjusted R² in logistic 
> regression is AIC. AIC is the measure of fit which penalizes model for the number of 
> model coefficients. Therefore, we always prefer model with minimum AIC value.
> 
> 2. `Null Deviance and Residual Deviance` – Null Deviance indicates the response predicted
>  by a model with nothing but an intercept. Lower the value, better the model. Residual 
>  deviance indicates the response predicted by a model on adding independent variables. 
>  Lower the value, better the model.
>  
> 3. `Confusion Matrix` – It is nothing but a tabular representation of Actual vs Predicted values.
>  This helps us to find the accuracy of the model and avoid overfitting. 
>  This is how it looks like:
>  ![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/confusion_table.png)
>  
>  ![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/confusion_matrix_formula.png)
>  
> 4. `ROC Curve: Receiver Operating Characteristic(ROC)` –  Summarizes the model’s performance 
> by evaluating the trade offs between true positive rate (sensitivity) and false positive 
> rate(1- specificity). For plotting ROC, it is advisable to assume p > 0.5 since we are more 
> concerned about success rate. ROC summarizes the predictive power for all possible values of 
> p > 0.5.  The area under curve (AUC), referred to as index of accuracy(A) or concordance index, 
> is a perfect performance metric for ROC curve. Higher the area under curve, better the prediction 
> power of the model. Below is a sample ROC curve. The ROC of a perfect predictive model has TP 
> equals 1 and FP equals 0. This curve will touch the top left corner of the graph.
>
From [24]:
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/roc_curve_graph.png)

> `Note`: For model performance, you can also consider likelihood function. 
> It is called so, because it selects the coefficient values which maximizes 
> the likelihood of explaining the observed data. It indicates goodness of fit 
> as its value approaches one, and a poor fit of the data as its value approaches zero.
> 
> -- <cite> Analytics Vidhya </cite>




## 1. Data Tendency:


According with google documentation (quoting):

> In general, the more training examples you have, the better your outcome.
> The amount of example data required also scales with the complexity of the
> problem you're trying to solve. You won't need as much data to get an accurate
>  binary classification model compared to a multi-class model because it's less 
>  complicated to predict one class from two rather than many.
>
> There's no perfect formula, but there are recommended minimum rows of example data:
> - Classification problem: 50 x the number features
> - Regression problem: 200 x the number of features

> -- <cite> Google Cloud </cite>





## 2. Accuracy 
`-> Compare real prediction vs ml prediction`

```math
accuracy = (correctly predicted class / total testing class) × 100%
```
Using: <br/>

`(TP)` True-Positive Rate = TP / TP + FN   <br/>
`(FP)` False-Positive Rate = FP / FP + TN  <br/>
`(TN)` True-Negative Rate = TN / TN + FP   <br/>
`(FN)` False-Negative Rate  = FN / FN + TP  <br/>

```math
accuracy = (TP + TN)/(TP + TN + FP + FN)
```

## 3. Precision 
Is the ratio of correct positive predictions out of all positive predictions made, or the accuracy of minority class predictions.

```math
precision = TruePositives / (TruePositives + FalsePositives)
```

## 4. Sensitivity 

```math
sensitivity = TP / TP + FN  
```
## 5. Specificity 

```math
specificity = TN / TN + FP  
```

## 6. Auto ML 

Considering Microsoft documentation:

### Types of AutoML: classify, regression, & forecast
### - Classification 
> Classification is a common machine learning task. Classification is a 
> type of supervised learning in which models learn using training data, 
> and apply those learnings to new data. Azure Machine Learning offers 
> featurizations specifically for these tasks, such as deep neural network 
> text featurizers for classification. Learn more about <a href="https://github.com/MicrosoftDocs/azure-docs/blob/master/articles/machine-learning/how-to-configure-auto-features.md#featurization" target="_top"> featurization options </a>.
> 
> The main goal of classification models is to predict which categories 
> new data will fall into based on learnings from its training data. 
> Common classification examples include fraud detection, handwriting 
> recognition, and object detection. Learn more and see an example at <a href="https://github.com/MicrosoftDocs/azure-docs/blob/master/articles/machine-learning/tutorial-first-experiment-automated-ml.md" target="_top"> Create a classification model with automated ML </a>.
>
> See examples of classification and automated machine learning in these 
> Python notebooks:
> 1.  <a href="https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/automated-machine-learning/classification-credit-card-fraud/auto-ml-classification-credit-card-fraud.ipynb" target="_top"> Fraud Detection </a>
> 2. <a href="https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/automated-machine-learning/classification-bank-marketing-all-features/auto-ml-classification-bank-marketing-all-features.ipynb" target="_top"> Marketing Prediction </a>
> 3. <a href="https://towardsdatascience.com/automated-text-classification-using-machine-learning-3df4f4f9570b" target="_top"> Newsgroup Data Classification </a>
> 
> -- <cite> Microsoft Docs </cite>

### - Regression
> Similar to classification, regression tasks are also a common 
> supervised learning task. Azure Machine Learning 
> <a href="https://github.com/MicrosoftDocs/azure-docs/blob/master/articles/machine-learning/how-to-configure-auto-features.md#featurization" target="_top"> offers featurizations specifically for these tasks </a>.

> Different from classification where predicted output values are categorical, 
> regression models predict numerical output values based on independent predictors. 
> In regression, the objective is to help establish the relationship among those independent
> predictor variables by estimating how one variable impacts the others. 
>  
>  For example, automobile price based on features like, gas mileage, 
>  safety rating, etc. Learn more and see an example of 
>  <a href="https://github.com/MicrosoftDocs/azure-docs/blob/master/articles/machine-learning/tutorial-auto-train-models.md" target="_top"> regression with automated machine learning </a>.
>
>
> See examples of regression and automated machine 
> learning for predictions in these Python notebooks: 
> <a href="https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/automated-machine-learning/regression-explanation-featurization/auto-ml-regression-explanation-featurization.ipynb" target="_top" > CPU Performance Prediction </a>
> 
> -- <cite> Microsoft Docs </cite>


### - Time-series forecasting
> Building forecasts is an integral part of any business, whether it's revenue,
> inventory, sales, or customer demand. You can use automated ML to combine 
> techniques and approaches and get a recommended, high-quality time-series forecast.
> Learn more with this how-to: 
> <a href="https://github.com/MicrosoftDocs/azure-docs/blob/master/articles/machine-learning/how-to-auto-train-forecast.md" target="_top" > automated machine learning for time series forecasting </a>.
>
> An automated time-series experiment is treated as a multivariate regression problem. 
> Past time-series values are "pivoted" to become additional dimensions for the regressor 
> together with other predictors. This approach, unlike classical time series methods, 
> has an advantage of naturally incorporating multiple contextual variables and their 
> relationship to one another during training. Automated ML learns a single, but often 
> internally branched model for all items in the dataset and prediction horizons. 
> More data is thus available to estimate model parameters and generalization to unseen 
> series becomes possible.
>
> Advanced forecasting configuration includes:
> 
> - Holiday detection and featurization
> - Time-series and DNN learners (Auto-ARIMA, Prophet, ForecastTCN)
> - Many models support through grouping
> - Rolling-origin cross validation
> - Configurable lags
> - Rolling window aggregate features
>
> See examples of regression and automated machine learning for predictions 
> in these Python notebooks: 
> 1. <a href="https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/automated-machine-learning/forecasting-orange-juice-sales/auto-ml-forecasting-orange-juice-sales.ipynb" target="_top" > Sales Forecasting </a>
> 2. <a href="https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/automated-machine-learning/forecasting-energy-demand/auto-ml-forecasting-energy-demand.ipynb" target="_top" > Demand Forecasting </a>
> 3. <a href="https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/automated-machine-learning/forecasting-beer-remote/auto-ml-forecasting-beer-remote.ipynb" target="_top" >  Beverage Production Forecast </a>
> 
> -- <cite> Microsoft Docs </cite>


### Ensemble models
> Automated machine learning supports ensemble models, which are enabled by default. 
> Ensemble learning improves machine learning results and predictive performance by 
> combining multiple models as opposed to using single models. The ensemble iterations 
> appear as the final iterations of your run. Automated machine learning uses both voting
> and stacking ensemble methods for combining models:
>
> - Voting: predicts based on the weighted average of predicted class probabilities 
> (for classification tasks) or predicted regression targets (for regression tasks).
> - Stacking: stacking combines heterogenous models and trains a meta-model based on 
> the output from the individual models. The current default meta-models are LogisticRegression for 
> classification tasks and ElasticNet for regression/forecasting tasks.
>
> The Caruana ensemble selection algorithm with sorted ensemble initialization is used 
> to decide which models to use within the ensemble. At a high level, this algorithm initializes 
> the ensemble with up to five models with the best individual scores, and verifies that these 
> models are within 5% threshold of the best score to avoid a poor initial ensemble. Then for 
> each ensemble iteration, a new model is added to the existing ensemble and the resulting score 
> is calculated. If a new model improved the existing ensemble score, the ensemble is 
> updated to include the new model.
>
> See the how-to for changing default ensemble settings in automated machine learning.
>
> -- <cite> Microsoft Docs </cite>


### Transform Strategies

|  Transform Strategies  |   Meaning  | Syntax     | 
|     ---    |        ---       |         ---      |
| Constant   |   Fill missing values in the target column or features, with zeroes    |   featurization_config.add_transformer_params('Imputer', target_columns, {"strategy": "constant", "fill_value": 0})  | 
| Median     | Fill mising values in the target column with median value                |      featurization_config.add_transformer_params('Imputer', target_columns, {"strategy": "median"}) | 
| Most Frequent  |      Fill mising values in the target column with most frequent value         |        featurization_config.add_transformer_params('Imputer', target_columns, {"strategy": "most_frequent"})           | 


## References
[1] From https://searchcloudcomputing.techtarget.com/definition/Microsoft-Azure-Machine-Learning#:~:text=Microsoft%20Azure%20Machine%20Learning%20is,through%20its%20Azure%20public%20cloud <br/>
[2] From https://www.bbva.com/es/machine-learning-que-es-y-como-funciona/ <br/>
[3] From https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9 <br/>
[4] From https://machinelearningmastery.com/precision-recall-and-f-measure-for-imbalanced-classification/ <br/>
[5] From https://cloud.google.com/automl-tables/docs/beginners-guide <br/>
[6] From https://github.com/MicrosoftDocs/azure-docs/blob/master/articles/machine-learning/how-to-auto-train-forecast.md <br/>
[7] From https://github.com/MicrosoftDocs/azure-docs/blob/master/articles/machine-learning/concept-automated-ml.md <br/>
[8] From https://www.datarobot.com/wiki/target/ <br/>
[9] From https://ai.plainenglish.io/different-types-of-machine-learning-algorithms-28974016e108 <br/>
[10] From https://www.researchgate.net/post/Class-imbalanced-dataset-for-Machine-Learning-how-to-test-it <br/>
[11] From https://towardsdatascience.com/types-of-machine-learning-algorithms-you-should-know-953a08248861 <br/>
[12] From https://www.potentiaco.com/what-is-machine-learning-definition-types-applications-and-examples/ <br/>
[13] From https://corporatefinanceinstitute.com/resources/knowledge/other/decision-tree/ <br/>
[14] From https://developer.ibm.com/technologies/artificial-intelligence/articles/cc-models-machine-learning/ <br/>
[15] From https://www.javatpoint.com/linear-regression-vs-logistic-regression-in-machine-learning <br/>
[16] From https://www.javatpoint.com/unsupervised-machine-learning <br/>
[17] From https://www.kdnuggets.com/2018/03/5-things-reinforcement-learning.html <br/>
[18] From https://medium.com/@Medmain/learning-through-trial-and-error-f83ab6e591dd <br/>
[19] From https://machinelearningmastery.com/types-of-learning-in-machine-learning/ <br/>
[20] From https://machinelearningmastery.com/logistic-regression-for-machine-learning/ <br/>
[21] From https://www.javatpoint.com/linear-regression-vs-logistic-regression-in-machine-learning <br/>
[22] From https://www.softwaretestinghelp.com/types-of-machine-learning-supervised-unsupervised/ <br/>
[23] From https://www.analyticsvidhya.com/blog/2015/11/beginners-guide-on-logistic-regression-in-r/ <br/>
[24] From https://www.datasciencecentral.com/profiles/blogs/roc-curve-explained-in-one-picture <br/>
