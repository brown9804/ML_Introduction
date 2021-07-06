# [Notes] Machine Learning Load Data

----------

Costa Rica

Belinda Brown, belindabrownr04@gmail.com

Jan, 2021

----------

## Data Tendency:

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

## `READ SOURCE DATA`:

```python 
import pandas as pd

pd_df = pd.read_csv("./file_name.csv", sep=',', encoding='utf-8', engine='python',error_bad_lines=False)

print("Data Frame Shape:  ", pd_df.shape)
```

## `CLEANING & IDENTIFICATION `
It's important to consider lambda method:

> `lambda` is a keyword that returns a function object and does not create a 'name'. Whereas def creates name in the local namespace <br/>
> `lambda` functions are good for situations where you want to minimize lines of code as you can create function in one line of python code. 
> It is not possible using def <br/>
> > -- <cite> RSGB Business Consultan From [3] </cite>

### Types of Data for conversations 
From [4]:
![Dtypes](https://github.com/brown9804/ML_DS_path/blob/main/_docs/img/python_dtype.png) 

```python 
import numpy as np
import pandas as pd

# Cleaning 
## ----------------- Remove duplicates 
pd_without_duplicates = pd_df.drop_duplicates()
print("Data Frame Shape without duplicates: ",pd_without_duplicates.shape)
## ----------------- Remove null values 
pd_without_duplicates_and_nulls = pd_without_duplicates[pd_without_duplicates.origin.notnull()]
## ----------------- Filtering by important samples 
# Operators >, < ==, != 
filtered_df = pd_without_duplicates_and_nulls[pd_without_duplicates_and_nulls.apply(lambda x: x["columnName_1"] == 'Column_value_want_it' and x["columnName_2"] != 'No_want_it_value', axis=1)] 
## ----------------- Change DataTypes 
df_units_conversation = pd_without_duplicates_and_nulls.astype({'col_name_2':'float64', 'col_name_3':'float64'})
## ----------------- Data Manipulatation by column
df_units_conversation['Output_in_Another_column'] = df_units_conversation.apply(lambda x: (If_condition_happend_this_is_going_to_executed) if x.ConditionColumn != 'ConditionValue' else (do_this), axis=1)
# Differents ways 
# df_units_conversation['Output_in_Same_Column'] = df_units_conversation.apply(lambda x: x.Output_in_Same_Column if x.ConditionColumn != 'ConditionValue' else print("No changes"), axis=1)

# Mapping dataset
mapped_dataset = df_units_conversation.groupby('objects_to_classify')['Classification'].value_counts()
mapped_dataset.head(10) # print first 10 rows 

# Debug Mapped columns 
def debug_mapped_columns(target_column, numerical_columns, categorical_columns, exclude_columns):
# example of def categorical_columns = ['column_1','another_one']
  pd_mapped = pd.DataFrame(target_column + numerical_columns + categorical_columns + exclude_columns)
  # Debugger
  if pd_without_duplicates.shape[1] != pd_mapped.shape[0]:
      print("Need to review columns, something is missing since: \n Initial df size without duplicates", pd_without_duplicates.shape[1], " vs Debug Mapped df: ", pd_mapped.shape[0])
  else:
      print("All mapped")    
```

## `Identification of datasets`

``` python 
# Identify columns class from mapped dataset
target_column = mapped_dataset['column_name_0', 'column_name_1']
numerical_columns = mapped_dataset['column_name_2', 'column_name_5']
categorical_columns = mapped_dataset['column_name_6', 'column_name_7']
exclude_columns = mapped_dataset['column_name_8', 'column_name_9']
## ----------------- Debug Mapped columns 
debug_mapped_df = debug_mapped_columns(target_column, numerical_columns, categorical_columns, exclude_columns)

# Describe all data -> data characteristics 
debug_mapped_df.describe(include='all')
```

### `→ For Supervised:`
Classification example based on [5]:

```python
from sklearn.model_selection import train_test_split
# Split into inputs and outputs the dataset
X, y = debug_mapped_df[:, :-1], debug_mapped_df[:, -1]
print(X.shape, y.shape)
# Split into Train Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```


### `→ For Unsupervised:`
KNeighborsClassifier example based on [6]:

```python
n_samples = 6
x = debug_mapped_df.iloc[:, :2]
y = debug_mapped_df.iloc[:, -1]

model_KNeighborsClassifier = KNeighborsClassifier(n_samples, weights='distance')
model_KNeighborsClassifier.fit(x, y)
```


### `→ For Reinforcement:`
Data-driven reinforcement learning (RL), with PyBullet example based on [7]:

```python
# d4rl dataset, is the first standardized dataset for this domain
import gym
import d4rl_pybullet

# dataset will be automatically downloaded into ~/.d4rl/datasets
env = gym.make('hopper-bullet-mixed-v0')

# interaction with its environment
env.reset()
env.step(env.action_space.sample())

# access to the dataset
dataset = env.get_dataset()
dataset['observations'] # observation data in N x dim_observation
dataset['actions'] # action data in N x dim_action
dataset['rewards'] # reward data in N x 1
dataset['terminals'] # terminal flags in N x 1
```

Data-driven reinforcement learning (RL), with  Atari datasets released by Google example based on [7]:

``` python
import gym
import d4rl_atari

env = gym.make('breakout-mixed-v0', stack=False) # -v{0, 1, 2, 3, 4} for datasets with the other random seeds

# interaction with its environment through dopamine-style Atari wrapper
env.reset() # observation is resized to 1x84x84
env.step(env.action_space.sample())

# dataset will be automatically downloaded into ~/.d4rl/datasets/[GAME]/[INDEX]/[EPOCH]
dataset = env.get_dataset()
dataset['observations'] # observation data in 1000000x1x84x84
dataset['actions'] # action data in 1M
dataset['rewards'] # reward data in 1M
dataset['terminals'] # terminal flags in 1M
```

## * References 
[1] From https://re-thought.com/pandas-value_counts/ <br/>
[2] From https://www.listendata.com/2019/07/how-to-filter-pandas-dataframe.html <br/>
[3] From https://www.listendata.com/2019/04/python-lambda-function.html <br/>
[4] From https://pbpython.com/pandas_dtypes.html <br/>
[5] From https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/ <br/>
[6] From https://github.com/SidsMav2000/TSF-Task-2---Data-Science-Business-Analytics-Tasks/blob/main/irisPrediction/Prediction.py <br/>
[7] From https://towardsdatascience.com/introducing-completely-free-datasets-for-data-driven-deep-reinforcement-learning-a51e9bed85f9 <br/>
