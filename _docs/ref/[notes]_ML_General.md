# [Notes] Machine Learning 

----------

Costa Rica

Belinda Brown, belindabrownr04@gmail.com

Jan, 2021

----------

It is a branch of artificial intelligence that allows machines to learn without being expressly programmed to do so. The idea is to identify the patterns between the data to make predictions. That is, the relationships between the columns are analyzed.

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


## Dictionary
`Target Variable`: The target variable of a dataset is the feature of a dataset about which you want to gain a deeper understanding. [8]

## 3 main Types of Learning:

|   Type  | Category | Definition |
|---|---|---|
| Supervised | Classification and Regression | |
| Unsupervised | Clustering and  | |
| Reinforcement | *  | |




From [10]:
<br/>


### Topic #1 Supervised Learning 

|   Use  | % | 
|---|---|
| Training | 80 | 
| Validation | 20 |

#### 1.1 Decision Trees

##### 1.1.1 Classification Trees

##### 1.1.1 Regression Trees



#### 1.2 Logictic Regression

#### 1.3 Linear Regression


### Topic #2 Unsupervised Learning 





### Topic #3 Reinforcement Learning 










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

