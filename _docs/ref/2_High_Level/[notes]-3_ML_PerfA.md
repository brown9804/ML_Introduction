# [Notes] Machine Learning Performance Analysis 

----------

Costa Rica

Belinda Brown, belindabrownr04@gmail.com

Jan, 2021

----------

## *  How to choose a Performance Metric


Based on [10]:
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/How-to-Choose-a-Metric-for-Imbalanced-Classification-latest.png)


## * Averaging Methods provided by Scikit-learn 

Based on [27]: 

|   Method  | Definition |
|---|---|
| Macro  | Calculate the metric for each class and take the unweighted average. `Tend`: If `classes` have `different numbers of samples`, it might be more informative to use a macro average where minority classes are given equal weighting to majority classes |
| Micro  | Calculate the metric globally by counting the total true positives, false negatives, and false positives (independent of classes) |
| Weighted | Calculate the metric for each class and take the weighted average based on the number of samples per class. |


### 1. Classification metrics:

Based on [28]:

Automated ML calculates performance metrics for each classification model generated for your experiment. These metrics are based on the scikit learn implementation.:

Considering:

`(TP)` True-Positive Rate = TP / TP + FN   <br/>
`(FP)` False-Positive Rate = FP / FP + TN  <br/>
`(TN)` True-Negative Rate = TN / TN + FP   <br/>
`(FN)` False-Negative Rate  = FN / FN + TP  <br/>

Based on [28]:

|   Metric  | Formula | Definition |
|---|---|---|
|matthews\_correlation | <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;{\displaystyle&space;\mathrm&space;{MCC}&space;={\frac&space;{\mathrm&space;{TP}&space;\times&space;\mathrm&space;{TN}&space;-\mathrm&space;{FP}&space;\times&space;\mathrm&space;{FN}&space;}{\sqrt&space;{(\mathrm&space;{TP}&space;&plus;\mathrm&space;{FP}&space;)(\mathrm&space;{TP}&space;&plus;\mathrm&space;{FN}&space;)(\mathrm&space;{TN}&space;&plus;\mathrm&space;{FP}&space;)(\mathrm&space;{TN}&space;&plus;\mathrm&space;{FN}&space;)}}}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;{\displaystyle&space;\mathrm&space;{MCC}&space;={\frac&space;{\mathrm&space;{TP}&space;\times&space;\mathrm&space;{TN}&space;-\mathrm&space;{FP}&space;\times&space;\mathrm&space;{FN}&space;}{\sqrt&space;{(\mathrm&space;{TP}&space;&plus;\mathrm&space;{FP}&space;)(\mathrm&space;{TP}&space;&plus;\mathrm&space;{FN}&space;)(\mathrm&space;{TN}&space;&plus;\mathrm&space;{FP}&space;)(\mathrm&space;{TN}&space;&plus;\mathrm&space;{FN}&space;)}}}}" title="{\displaystyle \mathrm {MCC} ={\frac {\mathrm {TP} \times \mathrm {TN} -\mathrm {FP} \times \mathrm {FN} }{\sqrt {(\mathrm {TP} +\mathrm {FP} )(\mathrm {TP} +\mathrm {FN} )(\mathrm {TN} +\mathrm {FP} )(\mathrm {TN} +\mathrm {FN} )}}}}" /></a> | Matthews correlation coefficient is a balanced measure of accuracy, which can be used even if one class has many more samples than another. A coefficient of 1 indicates perfect prediction, 0 random prediction, and -1 inverse prediction. Objective: `Closer to 1 the better`. Range: [-1, 1] | 
| sensitivity |<a href="https://www.codecogs.com/eqnedit.php?latex=sensitivity&space;=&space;\frac{TP&space;}{&space;(TP&space;&plus;&space;FN)&space;}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?sensitivity&space;=&space;\frac{TP&space;}{&space;(TP&space;&plus;&space;FN)&space;}" title="sensitivity = \frac{TP }{ (TP + FN) }" /></a> | Is the metric that evaluates a model’s ability to predict true positives of each available category |
| specificity | <a href="https://www.codecogs.com/eqnedit.php?latex=specificity&space;=&space;\frac{TN&space;}{&space;(TN&space;&plus;&space;FP)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?specificity&space;=&space;\frac{TN&space;}{&space;(TN&space;&plus;&space;FP)}" title="specificity = \frac{TN }{ (TN + FP)}" /></a>  | Determines a model’s ability to predict if an observation does not belong to a specific category |
| accuracy |  <a href="https://www.codecogs.com/eqnedit.php?latex=accuracy&space;=&space;\frac{correctly&space;predicted&space;class&space;}{total&space;testing&space;class}&space;*100%" target="_blank"><img src="https://latex.codecogs.com/gif.latex?accuracy&space;=&space;\frac{correctly&space;predicted&space;class&space;}{total&space;testing&space;class}&space;*100%" title="accuracy = \frac{correctly predicted class }{total testing class} *100%" /></a> or <a href="https://www.codecogs.com/eqnedit.php?latex=accuracy&space;=&space;\frac{(TP&space;&plus;&space;TN)}{(TP&space;&plus;&space;TN&space;&plus;&space;FP&space;&plus;&space;FN)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?accuracy&space;=&space;\frac{(TP&space;&plus;&space;TN)}{(TP&space;&plus;&space;TN&space;&plus;&space;FP&space;&plus;&space;FN)}" title="accuracy = \frac{(TP + TN)}{(TP + TN + FP + FN)}" /></a>| `Compare real prediction vs ml prediction`. Accuracy is the ratio of predictions that exactly match the true class labels. Objective: `Closer to 1 the better`. Range: [0, 1]|
| precision | <a href="https://www.codecogs.com/eqnedit.php?latex=precision&space;=&space;\frac{TruePositives&space;}{&space;(TruePositives&space;&plus;&space;FalsePositives)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?precision&space;=&space;\frac{TruePositives&space;}{&space;(TruePositives&space;&plus;&space;FalsePositives)}" title="precision = \frac{TruePositives }{ (TruePositives + FalsePositives)}" /></a> | Is the ratio of correct positive predictions out of all positive predictions made, or the accuracy of minority class predictions. Objective: `Closer to 1 the better`.Range: [0, 1]. `precision_score_macro`, `precision_score_micro`, `precision_score_weighted` |
| recall | <a href="https://www.codecogs.com/eqnedit.php?latex=Recall&space;=&space;\frac{TP&space;}{(TP&space;&plus;&space;FN)&space;}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Recall&space;=&space;\frac{TP&space;}{(TP&space;&plus;&space;FN)&space;}" title="Recall = \frac{TP }{(TP + FN) }" /></a>| Is the proportion of actual positives that was identified correctly. Objective: `Closer to 1 the better`.Range: [0, 1] `recall_score_macro`, `recall_score_micro`, `recall_score_weighted` |
|F-Measure| <a href="https://www.codecogs.com/eqnedit.php?latex=F-Measure&space;=&space;\frac{(2&space;*&space;Precision&space;*&space;Recall)&space;}{&space;(Precision&space;&plus;&space;Recall)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F-Measure&space;=&space;\frac{(2&space;*&space;Precision&space;*&space;Recall)&space;}{&space;(Precision&space;&plus;&space;Recall)}" title="F-Measure = \frac{(2 * Precision * Recall) }{ (Precision + Recall)}" /></a> |Gives more weight to precision and less to recall. Fbeta-measure provides a configurable version of the F-measure to give more or less attention to the precision and recall measure when calculating a single score. Objective: `Closer to 1 the better`.Range: [0, 1]. `f1_score_macro`, `f1_score_micro`, `f1_score_weighted` |
|AUC | <a href="https://www.codecogs.com/eqnedit.php?latex=\int_{x_0}^{x_n}&space;ROC&space;dx" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\int_{x_0}^{x_n}&space;ROC&space;dx" title="\int_{x_0}^{x_n} ROC dx" /></a> | AUC is the Area under the Receiver Operating Characteristic Curve. Objective: `Closer to 1 the better`. Range: [0, 1]. `AUC_macro`, `AUC_micro`, `AUC_weighted` |
| average_precision | <a href="https://www.codecogs.com/eqnedit.php?latex=AP&space;=\sum_{n}^{}(R_n&space;-&space;R_{n-1})*P_n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?AP&space;=\sum_{n}^{}(R_n&space;-&space;R_{n-1})*P_n" title="AP =\sum_{n}^{}(R_n - R_{n-1})*P_n" /></a> | Average precision summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight. Objective: `Closer to 1 the better`.Range: [0, 1]. `average_precision_score_macro`, `average_precision_score_micro`, `average_precision_score_weighted` |
|balanced_accuracy | <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;balanced\_accuracy&space;=&space;\frac{Recall_n}{n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;balanced\_accuracy&space;=&space;\frac{Recall_n}{n}" title="balanced\_accuracy = \frac{Recall_n}{n}" /></a> | Balanced accuracy is the arithmetic mean of recall for each class.Objective: `Closer to 1 the better`.Range: [0, 1]  |
|log\_loss  | <a href="https://www.codecogs.com/eqnedit.php?latex=L_{log}(y,p)&space;=&space;-(ylog(p)&space;&plus;&space;(1-y)*log(1-p))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L_{log}(y,p)&space;=&space;-(ylog(p)&space;&plus;&space;(1-y)*log(1-p))" title="L_{log}(y,p) = -(ylog(p) + (1-y)*log(1-p))" /></a> |This is the loss function used in (multinomial) logistic regression and extensions of it such as neural networks, defined as the negative log-likelihood of the true labels given a probabilistic classifier's predictions. Objective: `Closer to 0 the better`. Range: [0, inf) |
|norm\_macro\_recall | <a href="https://www.codecogs.com/eqnedit.php?latex=norm\_macro\_recall&space;=&space;\frac{recall\_score\_macro&space;-&space;R}{&space;1&space;-&space;R}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?norm\_macro\_recall&space;=&space;\frac{recall\_score\_macro&space;-&space;R}{&space;1&space;-&space;R}" title="norm\_macro\_recall = \frac{recall\_score\_macro - R}{ 1 - R}" /></a> where, R is the expected value of recall_score_macro for random predictions. <br/> R = 0.5 for  binary classification. <br/> R = (1 / C) for C-class classification problems. | Normalized macro recall is recall macro-averaged and normalized, so that random performance has a score of 0, and perfect performance has a score of 1. Objective: `Closer to 1 the better`.Range: [0, 1]|
|weighted_accuracy | <a href="https://www.codecogs.com/eqnedit.php?latex=weighted\_accuracy&space;=&space;\frac{accuracy\_n}{n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?weighted\_accuracy&space;=&space;\frac{accuracy\_n}{n}" title="weighted\_accuracy = \frac{accuracy\_n}{n}" /></a> | Weighted accuracy is accuracy where each sample is weighted by the total number of samples belonging to the same class. Objective: `Closer to 1 the better`.Range: [0, 1] |

### 2. Regression/forecasting metrics:
Considering:

`(TP)` True-Positive Rate = TP / TP + FN   <br/>
`(FP)` False-Positive Rate = FP / FP + TN  <br/>
`(TN)` True-Negative Rate = TN / TN + FP   <br/>
`(FN)` False-Negative Rate  = FN / FN + TP  <br/>

Based on [28]:

|   Metric   | Definition |
|---|---|
explained\_variance | Explained variance measures the extent to which a model accounts for the variation in the target variable. It is the percent decrease in variance of the original data to the variance of the errors. When the mean of the errors is 0, it is equal to the coefficient of determination (see r2_score below). Objective: Closer to 1 the better. Range: (-inf, 1] |
|mean\_absolute\_error| Mean absolute error is the expected value of absolute value of difference between the target and the prediction. Objective: Closer to 0 the better. Range: [0, inf). Types: `mean_absolute_error`, `normalized_mean_absolute_error`, the mean_absolute_error divided by the range of the data. |
|mean\_absolute\_percentage\_error | Mean absolute percentage error (MAPE) is a measure of the average difference between a predicted value and the actual value. Objective: Closer to 0 the better. Range: [0, inf). |
|median\_absolute\_error | Median absolute error is the median of all absolute differences between the target and the prediction. This loss is robust to outliers. Objective: Closer to 0 the better. Range: [0, inf). Types: `median_absolute_error`, `normalized_median_absolute_error`: the median_absolute_error divided by the range of the data. |
|r2_score | R<sup>2</sup> (the coefficient of determination) measures the proportional reduction in mean squared error (MSE) relative to the total variance of the observed data. Objective: Closer to 1 the better. Range: [-1, 1]. Note: R2 often has the range (-inf, 1]. The MSE can be larger than the observed variance, so R<sup>2</sup> can have arbitrarily large negative values, depending on the data and the model predictions. Automated ML clips reported R<sup>2</sup> scores at -1, so a value of -1 for R<sup>2</sup> likely means that the true R<sup>2</sup> score is less than -1. Consider the other metrics values and the properties of the data when interpreting a negative R<sup>2</sup> score.|
|root_mean_squared_error	| 	Root mean squared error (RMSE) is the square root of the expected squared difference between the target and the prediction. For an unbiased estimator, RMSE is equal to the standard deviation. Objective: Closer to 0 the better. Range: [0, inf). Types: `root_mean_squared_error`, `normalized_root_mean_squared_error`: the root_mean_squared_error divided by the range of the data. |
| root_mean_squared_log_error | Root mean squared log error is the square root of the expected squared logarithmic error. Objective: Closer to 0 the better. Range: [0, inf). Types: `root_mean_squared_log_error`, `normalized_root_mean_squared_log_error`: the root_mean_squared_log_error divided by the range of the data. |
| spearman_correlation | Spearman correlation is a nonparametric measure of the monotonicity of the relationship between two datasets. Unlike the Pearson correlation, the Spearman correlation does not assume that both datasets are normally distributed. Like other correlation coefficients, Spearman varies between -1 and 1 with 0 implying no correlation. Correlations of -1 or 1 imply an exact monotonic relationship. Spearman is a rank-order correlation metric meaning that changes to predicted or actual values will not change the Spearman result if they do not change the rank order of predicted or actual values. Objective: Closer to 1 the better. Range: [-1, 1] | 

### 3. Specific Model - Perfomance Analysis:

Black-Box Models for interpreting your models regarding feature importance:

`Generalised Linear Models`:
> Generalised Linear Models (GLM’s) are all based on the following principle:
> if you take a linear combination of your features x with the model weights w,
> and feed the result through a squash function f, you can use it to predict a wide 
> variety of response variables. Most common applications for GLM’s are regression 
> (linear regression), classification (logistic regression) or modelling Poisson 
> processes (Poisson regression). The weights that are obtained after training are 
> a direct proxy of feature importance and they provide very concrete interpretation 
> of the model internals. e.g. when building a text classifier you can plot the most 
> important features and `verify whether the model is overfitting on noise`. If the most 
> important words do not correspond to your intuition (e.g. names or stopwords), 
> it probably means that the model is fitting to noise in the dataset and 
> it won’t perform well on new data. <br/>

`Random forest and SVM’s`:
> Even non-linear models such as tree based models (e.g. Random Forest) 
> also allow to obtain information on the feature importance. In Random 
> Forest, feature importance comes for free when training a model, so it is a 
> great way to verify initial hypotheses and identify ‘what’ the model is learning. 
> The weights in kernel based approaches such as SVM’s are often not a very 
> good proxy of feature importance. The advantage of kernel methods is that you are 
> able to `capture non-linear relations between variables by projecting the features
> into kernel space`. On the other hand, just looking at the weights as feature
> importance does not do justice to the feature interaction. <br/>

`Deep learning`:
> Deep learning models are notorious for their un-interpretability due to 
> the shear number of parameters and the complex approach to extracting and 
> combining features. As this class of models is able to obtain state-of-the-art 
> performance on a lot of tasks, a lot of research is focused on linking
> model predictions to the inputs. The two main approaches are either 
> gradient-based or attention-based.
> In `gradient-based methods`, the gradients of the target concept calculated in 
> a` backward pass are used to produce a map that highlights the important regions 
> in the input for predicting the target concept`. This is typically applied in 
> the context of `computer vision`. <br/>
> `Attention-based` methods are typically used with `sequential data (e.g. text data)`. 
> In addition to the normal weights of the network, attention weights are trained that act as 
> ‘input gates’. `These attention weights determine how much each of the different elements in
> the final network output`. Besides interpretability, attention within the context of the e.g.
> text-based question-answering also leads to better results as the network 
> is able to ‘focus’ its attention. <br/>

`LIME`:
> Lime is a more general framework that aims to `make the predictions of 
> ‘any’ machine learning model more interpretable`. In order to remain model-independent, 
> LIME works by `modifying the input to the model locally`. So instead of trying to 
> understand the entire model at the same time, `a specific input instance is modified and 
> the impact on the predictions are monitored`. In the context of text classification, 
> this means that some of `the words are e.g. replaced, to determine which elements 
> of the input impact the predictions`.
> > -- <cite> Towards DataScience from [34] </cite>

#### * Evaluate Logistic regression model
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



> 4. `ROC Curve: Receiver Operating Characteristic(ROC)` –  Summarizes the model’s performance 
> by evaluating the trade offs between true positive rate (sensitivity) and false positive 
> rate(1- specificity). For plotting ROC, it is advisable to assume p > 0.5 since we are more 
> concerned about success rate. ROC summarizes the predictive power for all possible values of 
> p > 0.5.  The area under curve (AUC), referred to as index of accuracy(A) or concordance index, 
> is a perfect performance metric for ROC curve. Higher the area under curve, better the prediction 
> power of the model. Below is a sample ROC curve. The ROC of a perfect predictive model has TP 
> equals 1 and FP equals 0. This curve will touch the top left corner of the graph.


Based on [24]:
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
[29] From https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html <br/>
[30] From https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html <br/>
[31] From https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html <br/>
[32] From https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html <br/>
[33] From https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html <br/>
[34] From https://towardsdatascience.com/interpretability-in-machine-learning-70c30694a05f <br/>
