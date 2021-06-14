# [Notes] Machine Learning Performance Analysis 

----------

Costa Rica

Belinda Brown, belindabrownr04@gmail.com

Jan, 2021

----------

## *  How to choose a Performance Metric


From [10]:
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/How-to-Choose-a-Metric-for-Imbalanced-Classification-latest.png)

### 1. Specific Model - Perfomance Analysis:

#### 1.1 Evaluate Logistic regression model
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

### 2. Explainability metrics:

Considering:

`(TP)` True-Positive Rate = TP / TP + FN   <br/>
`(FP)` False-Positive Rate = FP / FP + TN  <br/>
`(TN)` True-Negative Rate = TN / TN + FP   <br/>
`(FN)` False-Negative Rate  = FN / FN + TP  <br/>



|   Performance Metric  | Formula | Definition |
|---|---|---|
| Accuracy | accuracy = (correctly predicted class / total testing class) × 100% or accuracy = (TP + TN)/(TP + TN + FP + FN)| Compare real prediction vs ml prediction |
| Precision | precision = TruePositives / (TruePositives + FalsePositives) | Is the ratio of correct positive predictions out of all positive predictions made, or the accuracy of minority class predictions.|
| Sensitivity | sensitivity = TP / TP + FN  | Is the metric that evaluates a model’s ability to predict true positives of each available category |
| Specificity | specificity = TN / TN + FP  | Determines a model’s ability to predict if an observation does not belong to a specific category |
|F-Measure| F-Measure = (2 * Precision * Recall) / (Precision + Recall) |Gives more weight to precision and less to recall. Fbeta-measure provides a configurable version of the F-measure to give more or less attention to the precision and recall measure when calculating a single score. |



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
[26] From https://machinelearningmastery.com/fbeta-measure-for-machine-learning/#:~:text=The%20F0.,false%20negatives%2C%20then%20the%20F0 <br/>
