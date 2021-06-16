# [Notes] Machine Learning Fit Model

----------

Costa Rica

Belinda Brown, belindabrownr04@gmail.com

Jan, 2021

----------


##  Auto ML 

Considering Microsoft documentation:
> Apply automated ML when you want Azure Machine Learning to train and tune a
>  model for you using the target metric you specify. Automated ML democratizes 
>  the machine learning model development process, and empowers its users, 
>  no matter their data science expertise, to identify an end-to-end machine 
>  learning pipeline for any problem.
>  
>  Data scientists, analysts, and developers across industries can use automated ML to:
>  
>  - Implement ML solutions without extensive programming knowledge
>  
>  - Save time and resources
>  
>  - Leverage data science best practices
>  
>  - Provide agile problem-solving

> -- <cite> Microsoft Docs From [27] </cite>

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

See the example of configuring an Automated Machine Learning Experiment from [26]:

```python
automl_settings = {
    "n_cross_validations": 3,
    "primary_metric": 'average_precision_score_weighted',
    "enable_early_stopping": True,
    "max_concurrent_iterations": 2, # This is a limit for testing purpose, please increase it as per cluster size
    "experiment_timeout_hours": 0.25, # This is a time limit for testing purposes, remove it for real use cases, this will drastically limit ablity to find the best model possible
    "verbosity": logging.INFO,
}

automl_config = AutoMLConfig(task = 'classification',
                             debug_log = 'automl_errors.log',
                             compute_target = compute_target,
                             training_data = training_data,
                             label_column_name = label_column_name,
                             **automl_settings
                            )
```




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


## Ensemble models
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

