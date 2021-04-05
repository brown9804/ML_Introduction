# [Notes] Machine Learning 

It is a branch of artificial intelligence that allows machines to learn without being expressly programmed to do so. The idea is to identify the patterns between the data to make predictions. That is, the relationships between the columns are analyzed.

## References
[1] From https://searchcloudcomputing.techtarget.com/definition/Microsoft-Azure-Machine-Learning#:~:text=Microsoft%20Azure%20Machine%20Learning%20is,through%20its%20Azure%20public%20cloud <br/>
[2] From https://www.bbva.com/es/machine-learning-que-es-y-como-funciona/ <br/>
[3] From https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9 <br/>
[4] From https://machinelearningmastery.com/precision-recall-and-f-measure-for-imbalanced-classification/ <br/>
[5] From https://cloud.google.com/automl-tables/docs/beginners-guide <br/>
[6] From https://github.com/MicrosoftDocs/azure-docs/blob/master/articles/machine-learning/how-to-auto-train-forecast.md

## 1. Data Tendency:

|   Use  | % | 
|---|---|
| Training | 80 | 
| Validation | 20 |


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


|  Transform Strategies  |   Meaning  | Syntax     | 
|     ---    |        ---       |         ---      |
| Constant   |   Fill missing values in the target column or features, with zeroes    |   featurization_config.add_transformer_params('Imputer', target_columns, {"strategy": "constant", "fill_value": 0})  | 
| Median     | Fill mising values in the target column with median value                |      featurization_config.add_transformer_params('Imputer', target_columns, {"strategy": "median"}) | 
| Most Frequent  |      Fill mising values in the target column with most frequent value         |        featurization_config.add_transformer_params('Imputer', target_columns, {"strategy": "most_frequent"})           | 

