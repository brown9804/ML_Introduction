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

### How AutoML works
From [27]:
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/automl_diagram.png)

> Using Azure Machine Learning, you can design and run your automated ML training experiments with these steps:
> 1. Identify the ML problem to be solved: classification, forecasting, or regression <br/>
> 2. Choose whether you want to use the Python SDK or the studio web experience: Learn about the parity between the Python SDK and studio web experience. <br/>
> - For limited or no code experience, try the Azure Machine Learning studio web experience at https://ml.azure.com <br/>
> - For Python developers, check out the Azure Machine Learning Python SDK <br/>
> 3. Specify the source and format of the labeled training data: Numpy arrays or Pandas dataframe <br/>
> 4. Configure the compute target for model training, such as your local computer, Azure Machine Learning Computes, remote VMs, or Azure Databricks. <br/>
> 5. Configure the automated machine learning parameters that determine how many iterations over different models, hyperparameter settings, advanced preprocessing/featurization, and what metrics to look at when determining the best model. <br/>
> 6. Submit the training run. <br/>
> 7. Review the results <br/>
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


## * References
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
[25] From https://aprendeconalf.es/docencia/estadistica/manual/regresion/ <br/>
[26] From https://towardsdatascience.com/hidden-tricks-for-running-automl-experiment-from-azure-machine-learning-sdk-915d4e3f840e <br/>
[27] From https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml
