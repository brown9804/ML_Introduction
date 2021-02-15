#--  --  --  -- Introduction to Importing Data in Python
# Used for Data Scientist Training Path 
#FYI it's a compilation of how to work
#with different commands.


### --------------------------------------------------------
# # ------>>>>> Exploring your working directory
# In order to import data into Python, you should first
#  have an idea of what files are in your working directory.
# IPython, which is running on DataCamp's servers, 
# has a bunch of cool commands, including its magic 
# commands. For example, starting a line with ! gives 
# you complete system shell access. This means that the 
# IPython magic command ! ls will display the contents of 
# your current directory. Your task is to use the IPython 
# magic command ! ls to check out the contents of your 
# current directory and answer the following question: 
# which of the following files is in your working directory?
# R/ moby_dick.txt


### --------------------------------------------------------
# # ------>>>>> Importing entire text files
# Open a file: file
file = open('moby_dick.txt', mode='r')
# Print it
print(file.read())
# Check whether file is closed
print(file.closed)
# Close file
file.close()
# Check whether file is closed
print(file.closed)


### --------------------------------------------------------
# # ------>>>>> Importing text files line by line
# Read & print the first 3 lines
with open('moby_dick.txt') as file:
    print(file.readline())
    print(file.readline())
    print(file.readline())


### --------------------------------------------------------
# # ------>>>>> Pop quiz: examples of flat files
# You're now well-versed in importing text files and
# you're about to become a wiz at importing flat files. 
# But can you remember exactly what a flat file is? Test
# your knowledge by answering the following question: 
# which of these file types below is NOT an example of a flat file?
# R/ A relational database (e.g. PostgreSQL).




### --------------------------------------------------------
# # ------>>>>>Pop quiz: what exactly are flat files?
# Which of the following statements about flat files is incorrect?
# Flat files consist of rows and each row is called a record.
# Flat files consist of multiple tables with structured 
# relationships between the tables.
# A record in a flat file is composed of fields or 
# attributes, each of which contains at most one item of information.
# Flat files are pervasive in data science.
# R/ Flat files consist of multiple tables with structured relationships between the tables.



### --------------------------------------------------------
# # ------>>>>>Why we like flat files and the Zen of Python
# In PythonLand, there are currently hundreds of Python 
# Enhancement Proposals, commonly referred to as PEPs. PEP8, for example,
# is a standard style guide for Python, written by our sensei Guido van
# Rossum himself. It is the basis for how we here at DataCamp ask our
# instructors to style their code. Another one of my favorites is PEP20, 
# commonly called the Zen of Python. Its abstract is as follows:
# Long time Pythoneer Tim Peters succinctly channels the BDFL's guiding 
# principles for Python's design into 20 aphorisms, only 19 of which have 
# been written down.
# If you don't know what the acronym BDFL stands for, I suggest that you 
# look here. You can print the Zen of Python in your shell by typing import
# this into it! You're going to do this now and the 5th aphorism (line) 
# will say something of particular interest.
# The question you need to answer is: what is the 5th aphorism of the Zen of Python?
# R/ -- > command: import this
# Flat is better than nested.




### --------------------------------------------------------
# # ------>>>>> Using NumPy to import flat files
# Import package
import numpy as np
import matplotlib.pyplot as plt
# Assign filename to variable: file
file = 'digits.csv'
# Load file as array: digits
digits = np.loadtxt(file, delimiter=',')
# Print datatype of digits
print(type(digits))
# Select and reshape a row
im = digits[21, 1:]
im_sq = np.reshape(im, (28, 28))
# Plot reshaped data (matplotlib.pyplot already loaded as plt)
plt.imshow(im_sq, cmap='Greys', interpolation='nearest')
plt.show()




### --------------------------------------------------------
# # ------>>>>> Customizing your NumPy import
# Import numpy
import numpy as np
# Assign the filename: file
file = 'digits_header.txt'
# Load the data: data
data = np.loadtxt(file, delimiter='\t', skiprows=1, usecols=[0, 2])
# Print data
print(data)



### --------------------------------------------------------
# # ------>>>>> Importing different datatypes
import numpy as np
import matplotlib.pyplot as plt
# Assign filename: file
file = 'seaslug.txt'
# Import file: data
data = np.loadtxt(file, delimiter='\t', dtype=str)
# Print the first element of data
print(data[0])
# Import data as floats and skip the first row: data_float
data_float = np.loadtxt(file, delimiter='\t', dtype=float, skiprows=1)
# Print the 10th element of data_float
print(data_float[9])
# Plot a scatterplot of the data
plt.scatter(data_float[:, 0], data_float[:, 1])
plt.xlabel('time (min.)')
plt.ylabel('percentage of larvae')
plt.show()




### --------------------------------------------------------
# # ------>>>>> Working with mixed datatypes (1)
# Much of the time you will need to import datasets which have 
# different datatypes in different columns; one column may contain 
# strings and another floats, for example. The function np.loadtxt()
#  will freak at this. There is another function, np.genfromtxt(), 
# which can handle such structures. If we pass dtype=None to it, it
#  will figure out what types each column should be.
# Import 'titanic.csv' using the function np.genfromtxt() as follows:
# data = np.genfromtxt('titanic.csv', delimiter=',', names=True, dtype=None)
# Here, the first argument is the filename, the second specifies the delimiter, 
# and the third argument names tells us there is a header. Because the data are 
# of different types, data is an object called a structured array. Because numpy 
# arrays have to contain elements that are all the same type, the structured array 
# solves this by being a 1D array, where each element of the array is a row of the 
# flat file imported. You can test this by checking out the array's shape in the 
# shell by executing np.shape(data).
# Accessing rows and columns of structured arrays is super-intuitive: to get the 
# ith row, merely execute data[i] and to get the column with name 'Fare', execute data['Fare'].
# After importing the Titanic data as a structured array (as per the instructions above), 
# print the entire column with the name Survived to the shell. What are the last 
# 4 values of this column?
# R/ 1,0,1,0



### --------------------------------------------------------
# # ------>>>>> Working with mixed datatypes (2)
# Assign the filename: file
file = 'titanic.csv'
# Import file using np.recfromcsv: d
d = np.recfromcsv(file)
# Print out first three entries of d
print(d[:3])


### --------------------------------------------------------
# # ------>>>>> Using pandas to import flat files as DataFrames (1)
# Import pandas as pd
import pandas as pd
# Assign the filename: file
file = 'titanic.csv'
# Read the file into a DataFrame: df
df = pd.read_csv(file)
# View the head of the DataFrame
print(df.head())


### --------------------------------------------------------
# # ------>>>>> Using pandas to import flat files as DataFrames (2)
# Assign the filename: file
file = 'digits.csv'
# Read the first 5 rows of the file into a DataFrame: data
data = pd.read_csv(file, nrows=5, header=None)
# Build a numpy array from the DataFrame: data_array
data_array = data.values
# Print the datatype of data_array to the shell
print(type(data_array))


### --------------------------------------------------------
# # ------>>>>> Customizing your pandas import
# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
# Assign filename: file
file = 'titanic_corrupt.txt'
# Import file: data
data = pd.read_csv(file, sep='\t', comment='#', na_values='Nothing')
# Print the head of the DataFrame
print(data.head())
# Plot 'Age' variable in a histogram
pd.DataFrame.hist(data[['Age']])
plt.xlabel('Age (years)')
plt.ylabel('count')
plt.show()


### --------------------------------------------------------
# # ------>>>>> Not so flat any more
# In Chapter 1, you learned how to use the IPython magic command ! 
# ls to explore your current working directory. You can 
# also do this natively in Python using the library os, which
#  consists of miscellaneous operating system interfaces.
# The first line of the following code imports the library os, 
# the second line stores the name of the current directory in a 
# string called wd and the third outputs the contents of the directory in a list to the shell.
# import os
# wd = os.getcwd()
# os.listdir(wd)
# Run this code in the IPython shell and answer the 
# following questions. Ignore the files that begin with .
# Check out the contents of your current directory and answer the following 
# questions:
#  (1) which file is in your directory and NOT an example of a flat file;
#  (2) why is it not a flat file?
# R/ battledeath.xlsx is not a flat because it is a spreadsheet consisting of many sheets, not a single table.


### --------------------------------------------------------
# # ------>>>>> Loading a pickled file
# Import pickle package
import pickle
# Open pickle file and load data: d
with open('data.pkl', 'rb') as file:
    d = pickle.load(file)
# Print d
print(d)
# Print datatype of d
print(type(d))


### --------------------------------------------------------
# # ------>>>>> Listing sheets in Excel files
# Import pandas
import pandas as pd
# Assign spreadsheet filename: file
file = 'battledeath.xlsx'
# Load spreadsheet: xls
xls = pd.ExcelFile(file)
# Print sheet names
print(xls.sheet_names)


### --------------------------------------------------------
# # ------>>>>> Importing sheets from Excel files
# Load a sheet into a DataFrame by name: df1
df1 = xls.parse('2004')
# Print the head of the DataFrame df1
print(df1.head())
# Load a sheet into a DataFrame by index: df2
df2 = xls.parse(0)
# Print the head of the DataFrame df2
print(df2.head())


### --------------------------------------------------------
# # ------>>>>> Customizing your spreadsheet import
# Import pandas
import pandas as pd
# Assign spreadsheet filename: file
file = 'battledeath.xlsx'
# Load spreadsheet: xl
xls = pd.ExcelFile(file)
# Parse the first sheet and rename the columns: df1
df1 = xls.parse(0, skiprows=[0], names=['Country', 'AAM due to War (2002)'])
# Print the head of the DataFrame df1
print(df1.head())
# Parse the first column of the second sheet and rename the column: df2
df2 = xls.parse(1, usecols=[0], skiprows=[0], names=['Country'])
# # Print the head of the DataFrame df2
print(df2.head())


### --------------------------------------------------------
# # ------>>>>> How to import SAS7BDAT
# How do you correctly import the function SAS7BDAT() from the package sas7bdat?
# R/ from sas7bdat import SAS7BDAT


### --------------------------------------------------------
# # ------>>>>> Importing SAS files
# Import sas7bdat package
from sas7bdat import SAS7BDAT
# Save file to a DataFrame: df_sas
with SAS7BDAT('sales.sas7bdat') as file:
    df_sas = file.to_data_frame()
# Print head of DataFrame
print(df_sas.head())
# Plot histogram of DataFrame features (pandas and pyplot already imported)
pd.DataFrame.hist(df_sas[['P']])
plt.ylabel('count')
plt.show()


### --------------------------------------------------------
# # ------>>>>>  Using read_stata to import Stata files
# The pandas package has been imported in the environment as pd and 
# the file disarea.dta is in your working directory. 
# The data consist of disease extents for several diseases in various
#  countries (more information can be found here).
# What is the correct way of using the read_stata() function to import
#  disarea.dta into the object df?
# R/ df = pd.read_stata('disarea.dta')


### --------------------------------------------------------
# # ------>>>>> Importing Stata files
# Import pandas
import pandas as pd
# Load Stata file into a pandas DataFrame: df
df = pd.read_stata('disarea.dta')
# Print the head of the DataFrame df
print(df.head())
# Plot histogram of one column of the DataFrame
pd.DataFrame.hist(df[['disa10']])
plt.xlabel('Extent of disease')
plt.ylabel('Number of coutries')
plt.show()


### --------------------------------------------------------
# # ------>>>>> Using File to import HDF5 files
# The h5py package has been imported in the
#  environment and the file LIGO_data.hdf5 is
#  loaded in the object h5py_file.
# What is the correct way of using the h5py function, 
# File(), to import the file in h5py_file into an object, 
# h5py_data, for reading only?
# R/ h5py_data = h5py.File(h5py_file, 'r')


### --------------------------------------------------------
# # ------>>>>> Using h5py to import HDF5 files
# Import packages
import numpy as np
import h5py
# Assign filename: file
file = 'LIGO_data.hdf5'
# Load file: data
data = h5py.File(file, 'r')
# Print the datatype of the loaded file
print(type(data))
# Print the keys of the file
for key in data.keys():
    print(key)


### --------------------------------------------------------
# # ------>>>>> Extracting data from your HDF5 file
# Get the HDF5 group: group
group = data['strain']
# Check out keys of group
for key in group.keys():
    print(key)
# Set variable equal to time series data: strain
strain = data['strain']['Strain'].value
# Set number of time points to sample: num_samples
num_samples = 10000
# Set time vector
time = np.arange(0, 1, 1/num_samples)
# Plot data
plt.plot(time, strain[:num_samples])
plt.xlabel('GPS Time (s)')
plt.ylabel('strain')
plt.show()


### --------------------------------------------------------
# # ------>>>>> Loading .mat files
# Import package
import scipy.io
# Load MATLAB file: mat
mat = scipy.io.loadmat('albeck_gene_expression.mat')
# Print the datatype type of mat
print(type(mat))


### --------------------------------------------------------
# # ------>>>>> The structure of .mat in Python
# Print the keys of the MATLAB dictionary
print(mat.keys())
# Print the type of the value corresponding to the key 'CYratioCyt'
print(type(mat['CYratioCyt']))
# Print the shape of the value corresponding to the key 'CYratioCyt'
print(np.shape(mat['CYratioCyt']))
# Subset the array and plot it
data = mat['CYratioCyt'][25, 5:]
fig = plt.figure()
plt.plot(data)
plt.xlabel('time (min.)')
plt.ylabel('normalized fluorescence (measure of expression)')
plt.show()


### --------------------------------------------------------
# # ------>>>>> Pop quiz: The relational model
# Which of the following is not part of the relational model?
# Each row or record in a table represents an instance of an entity type.
# Each column in a table represents an attribute or feature of an instance.
# Every table contains a primary key column, which has a unique entry for each row.
# A database consists of at least 3 tables.
# There are relations between tables.
# R/ ---> A database consists of at least 3 tables.


### --------------------------------------------------------
# # ------>>>>> Creating a database engine
# Import necessary module
from sqlalchemy import create_engine
# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')


### --------------------------------------------------------
# # ------>>>>> What are the tables in the database?
# Import necessary module
from sqlalchemy import create_engine
# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')
# Save the table names to a list: table_names
table_names = engine.table_names()
# Print the table names to the shell
print(table_names)


### --------------------------------------------------------
# # ------>>>>> The Hello World of SQL Queries!
# Import packages
from sqlalchemy import create_engine
import pandas as pd
# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')
# Open engine connection: con
con = engine.connect()
# Perform query: rs
rs = con.execute('SELECT * FROM Album')
# Save results of the query to DataFrame: df
df = pd.DataFrame(rs.fetchall())
# Close connection
con.close()
# Print head of DataFrame df
print(df.head())


### --------------------------------------------------------
# # ------>>>>> Customizing the Hello World of SQL Queries
# Open engine in context manager
# Perform query and save results to DataFrame: df
with engine.connect() as con:
    rs = con.execute('SELECT LastName, Title FROM Employee')
    df = pd.DataFrame(rs.fetchmany(size=3))
    df.columns = rs.keys()
# Print the length of the DataFrame df
print(len(df))
# Print the head of the DataFrame df
print(df.head())



### --------------------------------------------------------
# # ------>>>>> Filtering your database records using SQL's WHERE
# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')
# Open engine in context manager
# Perform query and save results to DataFrame: df
with engine.connect() as con:
    rs = con.execute('SELECT * FROM Employee WHERE EmployeeID >= 6')
    df = pd.DataFrame(rs.fetchall())
    df.columns = rs.keys()
# Print the head of the DataFrame df
print(df.head())



### --------------------------------------------------------
# # ------>>>>> Ordering your SQL records with ORDER BY
# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')
# Open engine in context manager
with engine.connect() as con:
    rs = con.execute('SELECT * FROM Employee ORDER BY BirthDate')
    df = pd.DataFrame(rs.fetchall())
    # Set the DataFrame's column names
    df.columns = rs.keys()
# Print head of DataFrame
print(df.head())


### --------------------------------------------------------
# # ------>>>>> Pandas and The Hello World of SQL Queries!
# Import packages
from sqlalchemy import create_engine
import pandas as pd
# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')
# Execute query and store records in DataFrame: df
df =  pd.read_sql_query('SELECT * FROM Album', engine)
# Print head of DataFrame
print(df.head())
# Open engine in context manager and store query result in df1
with engine.connect() as con:
    rs = con.execute("SELECT * FROM Album")
    df1 = pd.DataFrame(rs.fetchall())
    df1.columns = rs.keys()
# Confirm that both methods yield the same result
print(df.equals(df1))



### --------------------------------------------------------
# # ------>>>>> Pandas for more complex querying
# Import packages
from sqlalchemy import create_engine
import pandas as pd
# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')
# Execute query and store records in DataFrame: df
df = pd.read_sql_query('SELECT * FROM Employee WHERE EmployeeId >= 6 ORDER BY BirthDate', engine)
# Print head of DataFrame
print(df.head())


### --------------------------------------------------------
# # ------>>>>> The power of SQL lies in relationships between tables: INNER JOIN
# Open engine in context manager
# Perform query and save results to DataFrame: df
with engine.connect() as con:
    rs = con.execute("SELECT Title, Name FROM Album INNER JOIN Artist on Album.ArtistID = Artist.ArtistID")
    df = pd.DataFrame(rs.fetchall())
    df.columns = rs.keys()
# Print head of DataFrame df
print(df.head())


### --------------------------------------------------------
# # ------>>>>> Filtering your INNER JOIN
# Execute query and store records in DataFrame: df
df = pd.read_sql_query('SELECT * FROM PlaylistTrack INNER JOIN Track on PlaylistTrack.TrackId = Track.TrackID WHERE Milliseconds < 250000', engine)
# Print head of DataFrame
print(df.head())