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


## * Averaging Methods provided by Scikit-learn 

From [27]: 

|   Method  | Definition |
|---|---|
| Macro  | Calculate the metric for each class and take the unweighted average. `Tend`: If `classes` have `different numbers of samples`, it might be more informative to use a macro average where minority classes are given equal weighting to majority classes |
| Micro  | Calculate the metric globally by counting the total true positives, false negatives, and false positives (independent of classes) |
| Weighted | Calculate the metric for each class and take the weighted average based on the number of samples per class. |


### 1. Classification metrics:

From [28]:

Automated ML calculates performance metrics for each classification model generated for your experiment. These metrics are based on the scikit learn implementation.:

Considering:

`(TP)` True-Positive Rate = TP / TP + FN   <br/>
`(FP)` False-Positive Rate = FP / FP + TN  <br/>
`(TN)` True-Negative Rate = TN / TN + FP   <br/>
`(FN)` False-Negative Rate  = FN / FN + TP  <br/>

From [28]:

|   Metric  | Formula | Definition |
|---|---|---|
| Accuracy |  <a href="https://www.codecogs.com/eqnedit.php?latex=accuracy&space;=&space;\frac{correctly&space;predicted&space;class&space;}{total&space;testing&space;class}&space;*100%" target="_blank"><img src="https://latex.codecogs.com/gif.latex?accuracy&space;=&space;\frac{correctly&space;predicted&space;class&space;}{total&space;testing&space;class}&space;*100%" title="accuracy = \frac{correctly predicted class }{total testing class} *100%" /></a> or <a href="https://www.codecogs.com/eqnedit.php?latex=accuracy&space;=&space;\frac{(TP&space;&plus;&space;TN)}{(TP&space;&plus;&space;TN&space;&plus;&space;FP&space;&plus;&space;FN)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?accuracy&space;=&space;\frac{(TP&space;&plus;&space;TN)}{(TP&space;&plus;&space;TN&space;&plus;&space;FP&space;&plus;&space;FN)}" title="accuracy = \frac{(TP + TN)}{(TP + TN + FP + FN)}" /></a>| `Compare real prediction vs ml prediction`. Accuracy is the ratio of predictions that exactly match the true class labels. Objective: Closer to 1 the better. Range: [0, 1]|
| Precision | <a href="https://www.codecogs.com/eqnedit.php?latex=precision&space;=&space;\frac{TruePositives&space;}{&space;(TruePositives&space;&plus;&space;FalsePositives)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?precision&space;=&space;\frac{TruePositives&space;}{&space;(TruePositives&space;&plus;&space;FalsePositives)}" title="precision = \frac{TruePositives }{ (TruePositives + FalsePositives)}" /></a> | Is the ratio of correct positive predictions out of all positive predictions made, or the accuracy of minority class predictions.|
| Sensitivity |<a href="https://www.codecogs.com/eqnedit.php?latex=sensitivity&space;=&space;\frac{TP&space;}{&space;(TP&space;&plus;&space;FN)&space;}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?sensitivity&space;=&space;\frac{TP&space;}{&space;(TP&space;&plus;&space;FN)&space;}" title="sensitivity = \frac{TP }{ (TP + FN) }" /></a> | Is the metric that evaluates a model’s ability to predict true positives of each available category |
| Specificity | <a href="https://www.codecogs.com/eqnedit.php?latex=specificity&space;=&space;\frac{TN&space;}{&space;(TN&space;&plus;&space;FP)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?specificity&space;=&space;\frac{TN&space;}{&space;(TN&space;&plus;&space;FP)}" title="specificity = \frac{TN }{ (TN + FP)}" /></a>  | Determines a model’s ability to predict if an observation does not belong to a specific category |
| Recall | <a href="https://www.codecogs.com/eqnedit.php?latex=Recall&space;=&space;\frac{TP&space;}{(TP&space;&plus;&space;FN)&space;}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Recall&space;=&space;\frac{TP&space;}{(TP&space;&plus;&space;FN)&space;}" title="Recall = \frac{TP }{(TP + FN) }" /></a>| Is the proportion of actual positives that was identified correctly |
|F-Measure| <a href="https://www.codecogs.com/eqnedit.php?latex=F-Measure&space;=&space;\frac{(2&space;*&space;Precision&space;*&space;Recall)&space;}{&space;(Precision&space;&plus;&space;Recall)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F-Measure&space;=&space;\frac{(2&space;*&space;Precision&space;*&space;Recall)&space;}{&space;(Precision&space;&plus;&space;Recall)}" title="F-Measure = \frac{(2 * Precision * Recall) }{ (Precision + Recall)}" /></a> |Gives more weight to precision and less to recall. Fbeta-measure provides a configurable version of the F-measure to give more or less attention to the precision and recall measure when calculating a single score. |
|AUC | <a href="https://www.codecogs.com/eqnedit.php?latex=\int_{x_0}^{x_n}&space;ROC&space;dx" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\int_{x_0}^{x_n}&space;ROC&space;dx" title="\int_{x_0}^{x_n} ROC dx" /></a> | AUC is the Area under the Receiver Operating Characteristic Curve. Objective: Closer to 1 the better. Range: [0, 1]. AUC\_macro, AUC\_micro, AUC\_weighted |
| average_precision | |
| | |
| | |
| | |
| | |
| | |
| | |


### 2. Specific Model - Perfomance Analysis:

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
[27] From https://docs.microsoft.com/en-us/azure/machine-learning/how-to-understand-automated-ml <br/>
[28] From https://docs.microsoft.com/en-us/azure/machine-learning/how-to-understand-automated-ml#binary-vs-multiclass-classification-metrics <br/>
