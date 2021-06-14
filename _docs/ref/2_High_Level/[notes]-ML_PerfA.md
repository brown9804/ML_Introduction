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

#### 2.1 Accuracy 
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

#### 2.2 Precision 
Is the ratio of correct positive predictions out of all positive predictions made, or the accuracy of minority class predictions.

```math
precision = TruePositives / (TruePositives + FalsePositives)
```

#### 2.3 Sensitivity 

```math
sensitivity = TP / TP + FN  
```
#### 2.4 Specificity 

```math
specificity = TN / TN + FP  
```

