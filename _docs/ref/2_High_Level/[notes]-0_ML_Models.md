# [Notes] Machine Learning Models

----------

Costa Rica

Belinda Brown, belindabrownr04@gmail.com

Jan, 2021

----------

It is a branch of artificial intelligence that allows machines to learn without being expressly programmed to do so. The idea is to identify the patterns between the data to make predictions. That is, the relationships between the columns are analyzed.

## *  Dictionary
`Target Variable`: The target variable of a dataset is the feature of a dataset about which you want to gain a deeper understanding. [8]

## *  3 main Types of Learning:

|   Type  | Category | Definition |
|---|---|---|
| Supervised | Classification and Regression | This algorithms learns the input pattern and generates the expected output. We have expected output associated with our input data. [9] |
| Unsupervised | Clustering and Association | The task of machine is to group unsorted information according to similarities, patterns and differences without any prior training of data. [9]|
| Reinforcement | *  | Reinforcement learning works by putting the algorithm in a work environment with an interpreter and a reward system. In every iteration of the algorithm, the output result is given to the interpreter, which decides whether the outcome is favorable or not. [12] |

From [18]:
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/types_of_ml_based_in_objective.png)

From [14]
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/ml_types_input_output.png)

From [28]
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/ml_supervised_un.png)

From [11]
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/machine_learning_types.png)

### Type #1 Supervised Learning 

|   Use  | % | 
|---|---|
| Training | 80 | 
| Validation | 20 |

From [14]:
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/supervised_model_in_out.png)

From [19]:
<br/>

|   Category  | Definition | 
|---|---|
| Classification | Supervised learning problem that involves predicting a class label. [19]| 
| Regression | Supervised learning problem that involves predicting a numerical label. [19]|


From [15]:
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/regression_vs_classification_supervised_learning.png)


From [22]:
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/supervised_general_flow.png)

#### 1.1 Decision Trees
> A decision tree is a supervised learning method for classification. 
> Algorithms of this variety create trees that predict the result of an 
> input vector based on decision rules inferred from the features present 
> in the data. Decision trees are useful because they’re easy to visualize 
> so you can understand the factors that lead to a result.
>
> Two types of models exist for decision trees: <br/>
> Classification trees, where the target variable is a discrete value and the leaves represent class labels (as shown in the example tree), and <br/>
> Regression trees, where the target variable can take continuous values.
> 
> > -- <cite> From [14] IBM </cite>
> > 

From [14]:
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/typical_decision_tree.png)

##### 1.1.1 Classification Trees (Discrete/Binary - Yes/No types)

> A categorical variable decision tree includes categorical target 
> variables that are divided into categories. For example, the categories 
> can be yes or no. The categories mean that every stage of the decision 
> process falls into one of the categories, and there are no in-betweens.
> 
> > -- <cite> From [13] CFI </cite>


##### 1.1.1 Regression Trees (Continuous data types)

> A continuous variable decision tree is a decision tree with a continuous 
> target variable. For example, the income of an individual whose income is
> unknown can be predicted based on available information such as their 
> occupation, age, and other continuous variables.
> 
> > -- <cite> From [13] CFI </cite>

#### 1.2 Logistic Regression
From [21]:
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/linear_logistic_regression_graph.png)

Assumptions:

> `Binary Output Variable`: This might be obvious as we have already mentioned it, 
> but logistic regression is intended for binary (two-class) classification problems. 
> It will predict the probability of an instance belonging to the default class,
> which can be snapped into a 0 or 1 classification.
> 
> `Remove Noise`: Logistic regression assumes no error in the output variable (y), 
> consider removing outliers and possibly misclassified instances from your training data.
> 
> `Gaussian Distribution`: Logistic regression is a linear algorithm (with a non-linear transform 
> on output). It does assume a `linear relationship between the input variables with the output`.
> Data transforms of your input variables that better expose this linear relationship can result
> in a more accurate model. For example, you can use log, root, Box-Cox and other univariate 
> transforms to better expose this relationship.
> 
> `Remove Correlated Input`: Like linear regression, the model can overfit if you have multiple 
> highly-correlated inputs. Consider calculating the pairwise correlations between all inputs 
> and removing highly correlated inputs.
> 
> `Fail to Converge`: It is possible for the expected likelihood estimation process that learns 
> the coefficients to fail to converge. This can happen if there are many highly correlated inputs
> in your data or the data is very sparse (e.g. lots of zeros in your input data).

> > -- <cite> From [20] Machine Learning Mastery </cite>




#### 1.3 Linear Regression

From [21]:
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/supervised_logistic_vs_linear_regression.png)

### Type #2 Unsupervised Learning 

From [14]:
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/unsupervised_logic_data_in_out.png)


From [16]:
<br/>

|   Categegory  | Definition | 
|---|---|
| Clustering | Clustering is a method of grouping the objects into clusters such that objects with most similarities remains into a group and has less or no similarities with the objects of another group. Cluster analysis finds the commonalities between the data objects and categorizes them as per the presence and absence of those commonalities. [16]| 
| Association | An association rule is an unsupervised learning method which is used for finding the relationships between variables in the large database. It determines the set of items that occurs together in the dataset. Association rule makes marketing strategy more effective. Such as people who buy X item (suppose a bread) are also tend to purchase Y (Butter/Jam) item. [16]|

From [22]:
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/unsupervised_general_flow.png)

### Type #3 Reinforcement Learning 
From [14]:
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/reinforcement_logic_data_in_out.png)


From [17]:
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/reinforcement_flow_basic_1.png)


From [17]:
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/reinforcement_flow_basic_2.png)


From [22]:
<br/>
![](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/reinforcement_general_flow.png)


## * Data Tendency:


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
[27] From https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml <br/>
[28] From https://algorithmia.com/blog/how-machine-learning-works <br/>
