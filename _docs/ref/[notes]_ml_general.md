# [Notes] Machine Learning 

It is a branch of artificial intelligence that allows machines to learn without being expressly programmed to do so. The idea is to identify the patterns between the data to make predictions. That is, the relationships between the columns are analyzed.

## 1. Data Tendency:

|   Use  | % | 
|---|---|
| Training | 80 | 
| Validation | 20 |


## 2. Accuracy 
`-> Compare real prediction vs ml prediction`

```math
accuracy = (correctly predicted class / total testing class) × 100%
```

Using: <br/>

Sensitivity = TP / TP + FN   <br/>
Specificity = TN / TN + FP   <br/>
Precision = TP / TP + FP     <br/>
True-Positive Rate = TP / TP + FN   <br/>
False-Positive Rate = FP / FP + TN  <br/>
True-Negative Rate = TN / TN + FP   <br/>
False-Negative Rate = FN / FN + TP  <br/>

```math
accuracy = (TP + TN)/(TP + TN + FP + FN)
```
