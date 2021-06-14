# ----------

# Costa Rica

# Belinda Brown, belindabrownr04@gmail.com

# Jan, 2021

# ----------

# To getting a small sample of n rows from a cvs to another one 
# if you want another extension just change it.
import pandas as pd

getting_sample = pd.read_csv("./file_.csv")
df = pd.DataFrame(getting_sample)
print(df.head(5).to_csv("./getting_sample.csv"))

# To tranpose a excel file n rows
df_n5 = df.head(5)
transpose_sample = df_n5.T
print(transpose_sample.to_csv("./transpose_n5_sample.csv"))

#Count rows 
import csv
with open("./file_path", "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader) # no headers 
    n_total = -1 # because headers 
    for filas_completas_data in csv_reader:
        n_total = n_total + 1
print(n_total)

#Import Data Frame to csv
DataFrame.to_csv('file_name.csv', encoding='utf-8')
