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
```python 
# Cleaning 
pd_without_duplicates = pd_df.drop_duplicates()
print("Data Frame Shape without duplicates: ",pd_without_duplicates.shape)

# Mapping dataset
mapped_dataset = pd_without_duplicates.groupby('objects_to_classify')['Classification'].value_counts()
mapped_dataset.head(10) # print first 10 rows 

# Identify columns 
def Identify_columns(target_column, numerical_columns, categorical_columns, exclude_columns):
# example of def categorical_columns = ['column_1','another_one']
  pd_mapped = pd.DataFrame(target_column + numerical_columns + categorical_columns + exclude_columns)
  # Debugger
  if Origin_df.shape[1] != pd_mapped.shape[0]:
      print("Need to review columns, something is missing since: \n Initial df size", Origin_df.shape[1], " vs ", pd_mapped.shape[0])
  else:
      print("All mapped")
```

## * References 
[1] From https://re-thought.com/pandas-value_counts/ <br/>
[2] From  <br/>
