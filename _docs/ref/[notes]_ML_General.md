# [Notes] Machine Learning 

It is a branch of artificial intelligence that allows machines to learn without being expressly programmed to do so. The idea is to identify the patterns between the data to make predictions. That is, the relationships between the columns are analyzed.

## References
[1] From https://searchcloudcomputing.techtarget.com/definition/Microsoft-Azure-Machine-Learning#:~:text=Microsoft%20Azure%20Machine%20Learning%20is,through%20its%20Azure%20public%20cloud. <br/>
[2] From https://www.bbva.com/es/machine-learning-que-es-y-como-funciona/ <br/>
[3] From https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9 <br/>

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
