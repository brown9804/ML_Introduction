  #--  --  --  --  Introduction to Python
# Used for Data Scientist Training Path 
#FYI it's a compilation of how to work
#with different commands.

## -----> Calculations with variables
### --------------------------------------------------------
# Division
print(5 / 8)

#Addition
print(7 + 10)

### --------------------------------------------------------
# Addition, subtraction
print(5 + 5)
print(5 - 5)

# Multiplication, division, modulo, and exponentiation
print(3 * 5)
print(10 / 2)
print(18 % 7)
print(4 ** 2)

# How much is your $100 worth after 7 years?
savings = 100
factor = 1.10
years = 7
result = (savings * factor ** years)
print(result)

## -----> Variables and Types 
### --------------------------------------------------------
# Create a variable savings
savings = 100 

# Print out savings
print(savings)

### --------------------------------------------------------
# Create a variable savings
savings = 100

# Create a variable growth_multiplier
growth_multiplier = 1.1
years = 7
# Calculate result
result = savings*growth_multiplier**years

# Print out result
print(result)

### --------------------------------------------------------
# Create a variable desc
desc = "compound interest"

# Create a variable profitable
profitable = True

### --------------------------------------------------------
type(a)
#Out[1]: float
type(b) 
#Out[2]: str
type(c)
#Out[3]: bool

### --------------------------------------------------------
savings = 100
growth_multiplier = 1.1
desc = "compound interest"

# Assign product of growth_multiplier and savings to year1
year1 = (savings * growth_multiplier)

# Print the type of year1
print(type(year1))

# Assign sum of desc and desc to doubledesc
doubledesc = desc + desc

# Print out doubledesc
print(doubledesc)

### --------------------------------------------------------
# Definition of savings and result
savings = 100
result = 100 * 1.10 ** 7

# Fix the printout
print("I started with $" + str(savings) + " and now have $" + str(result) + ". Awesome!")

# Definition of pi_string
pi_string = "3.1415926"

# Convert pi_string into float: pi_float
pi_float = float(pi_string)

## -----> List
### --------------------------------------------------------
# area variables (in square meters)
hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50

# Create list areas
areas = [hall, kit, liv, bed, bath]

# Print areas
print(areas)

### --------------------------------------------------------
# area variables (in square meters)
hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50

# Adapt list areas
areas = [ "hallway", hall,"kitchen", kit, "living room", liv,"bedroom", bed, "bathroom", bath]

# Print areas
print(areas)

### --------------------------------------------------------
##A. [1, 3, 4, 2]
##B. [[1, 2, 3], [4, 5, 7]]
##C. [1 + 2, "a" * 5, 3]
## Three possible combinations of lists


### --------------------------------------------------------
# area variables (in square meters)
hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50

# house information as list of lists
house = [["hallway", hall],
         ["kitchen", kit],
         ["living room", liv],
         ["bedroom", bed],
         ["bathroom", bath]
]
# Print out house
print(house)

# Print out the type of house
print(type(house))


### --------------------------------------------------------
# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Print out second element from areas
print(areas[1])

count = 0
for i in areas:
    count= count+ 1
print("The total of elements in areas are:" + str(count))
# Print out last element from areas
print(areas[count-1])

# Print out the area of the living room
print(areas[5])

### --------------------------------------------------------
# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Sum of kitchen and bedroom area: eat_sleep_area
eat_sleep_area = areas[3] + areas[-3]

# Print the variable eat_sleep_area
print(eat_sleep_area)


### --------------------------------------------------------
# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Use slicing to create downstairs
downstairs = areas[0:6]

# Use slicing to create upstairs
upstairs = areas[6:]

# Print out downstairs and upstairs
print(downstairs)
print(upstairs)


### --------------------------------------------------------
# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Alternative slicing to create downstairs
downstairs = areas[:6]

# Alternative slicing to create upstairs
upstairs = areas[6:]

print(downstairs)
print(upstairs)


### --------------------------------------------------------
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]
house[-1][1]
# returns the 9.5
# A float: the bathroom area


### --------------------------------------------------------
# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

print("Areas before: \n" + str(areas))

# Correct the bathroom area
areas[-1] = 10.50

# Change "living room" to "chill zone"
areas[4] = "chill zone"

print("Areas after: \n" + str(areas))


### --------------------------------------------------------
# Create the areas list and make some changes
areas = ["hallway", 11.25, "kitchen", 18.0, "chill zone", 20.0,
         "bedroom", 10.75, "bathroom", 10.50]

# Add poolhouse data to areas, new list is areas_1
areas_1 = areas + ["poolhouse", 24.5]

# Add garage data to areas_1, new list is areas_2
areas_2 = areas_1 + ["garage", 15.45]

print(areas)
print(areas_1)
print(areas_2)


### --------------------------------------------------------
## how to Delete list elements
areas = ["hallway", 11.25, "kitchen", 18.0,
        "chill zone", 20.0, "bedroom", 10.75,
         "bathroom", 10.50, "poolhouse", 24.5,
         "garage", 15.45]

del(areas[-4:-2])
# output 
# ['hallway', 11.25, 'kitchen', 18.0, 
# 'chill zone', 20.0, 'bedroom', 10.75, 
# 'bathroom', 10.5, 'garage', 15.45]


### --------------------------------------------------------
### Inner workings of lists
# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Create areas_copy
areas_copy = list(areas)

# Change areas_copy
areas_copy[0] = 5.0

# Print areas
print(areas)


### --------------------------------------------------------
# Create variables var1 and var2
var1 = [1, 2, 3, 4]
var2 = True

# Print out type of var1
print(type(var1))

# Print out length of var1
print("The len is: " + str(len(var1)))

# Convert var2 to an integer: out2
print("Before "+ str(type(var2)))
## operation
out2 = int(var2)
## printing for debug 
print("After "+ str(type(out2)))



### --------------------------------------------------------
# Help!
# Maybe you already know the name of a 
# Python function, but you still have to figure out
#  how to use it. Ironically, you have to ask for 
#  information about a function with another function: 
#  help(). In IPython specifically, you can also use ? 
#  before the function name.

# To get help on the max() function, for example, 
# you can use one of these calls:

# help(max) ? max
# Use the Shell to open up the documentation on
#  complex(). Which of the following statements is true?

# R/ # complex() takes two arguments: real and imag. real is a required argument, imag is an optional argument.



### --------------------------------------------------------
## Multiple arguments
# Create lists first and second
first = [11.25, 18.0, 20.0]
second = [10.75, 9.50]

# Paste together first and second: full
full = first + second

# Sort full in descending order: full_sorted
full_sorted = sorted(full, reverse=True)

# Print out full_sorted
print(full_sorted)

### --------------------------------------------------------
## Methods  - String Methods
# string to experiment with: place
place = "poolhouse"

# Use upper() on place: place_up
place_up = place.upper()

# Print out place and place_up
print(place)
print(place_up)

# Print out the number of o's in place
print(place.count('o'))


### --------------------------------------------------------
## Methods  - List Methods - ex#0
# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Print out the index of the element 20.0
print(areas.index(20.0))

# Print out how often 9.50 appears in areas
print(areas.count(9.50))



### --------------------------------------------------------
## Methods  - List Methods - ex#1
# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Use append twice to add poolhouse and garage size
areas.append(24.5)
areas.append(15.45)

# Print out areas
print(areas)

# Reverse the orders of the elements in areas
areas.reverse()

# Print out areas
print(areas)


### --------------------------------------------------------
## import example 
# Definition of radius
r = 0.43

# Import the math package
import math 

# Calculate C
C = 2 * math.pi * r

# Calculate A
A = math.pi * (r**2)

# Build printout
print("Circumference: " + str(C))
print("Area: " + str(A))



### --------------------------------------------------------
## import selected
# Definition of radius
r = 192500

# Import radians function of math package
from math import radians

# Travel distance of Moon over 12 degrees. Store in dist.
phi = radians(12)
dist = r * phi

# Print out dist
print(dist)


### --------------------------------------------------------
## Suppose you want to use the function inv(), which is in
## the linalg subpackage of the scipy package. You want to
## be able to use this function as follows:
## R/ from scipy.linalg import inv as my_inv

### --------------------------------------------------------
## NUMPY
## -> NumPy Array
# Create list baseball
baseball = [180, 215, 210, 210, 188, 176, 209, 200]

# Import the numpy package as np
import numpy as np

# Create a numpy array from baseball: np_baseball
np_baseball = np.array(baseball)

# Print out type of np_baseball
print(type(np_baseball))

### --------------------------------------------------------
## Baseball players' height
# height is available as a regular list

# Import numpy
import numpy as np

# Create a numpy array from height_in: np_height_in
height_in = np.array(height_in)

# Print out np_height_in
print(height_in)

# Convert np_height_in to m: np_height_m
np_height_m = height_in * 0.0254

# Print np_height_m
print(np_height_m)

### --------------------------------------------------------
## Baseball player's BMI
# height and weight are available as regular lists

# Import numpy
import numpy as np

# Create array from height_in with metric units: np_height_m
np_height_m = np.array(height_in) * 0.0254

# Create array from weight_lb with metric units: np_weight_kg
np_weight_kg = np.array(weight_lb) * 0.453592

# Calculate the BMI: bmi
bmi = np_weight_kg / np_height_m ** 2

# Print out bmi
print(bmi)


### --------------------------------------------------------
## Lightweight baseball players
# height and weight are available as a regular lists

# Import numpy
import numpy as np

# Calculate the BMI: bmi
np_height_m = np.array(height_in) * 0.0254
np_weight_kg = np.array(weight_lb) * 0.453592
bmi = np_weight_kg / np_height_m ** 2

# Create the light array
light = np.array(bmi < 21)

# Print out light
print(light)

# Print out BMIs of all baseball players whose BMI is below 21
print(bmi[light])


### --------------------------------------------------------
## NumPy Side Effects
##Have a look at this line of code:
##np.array([True, 1, 2]) + np.array([3, 4, False])
##Can you tell which code chunk builds the exact same 
# Python object? The numpy package is already imported as 
# np, so you can start experimenting in the IPython 
# Shell straight away!
# R/  np.array([4, 3, 0]) + np.array([0, 2, 2])


### --------------------------------------------------------
## Subsetting NumPy Arrays
# height and weight are available as a regular lists
# Import numpy
import numpy as np

# Store weight and height lists as numpy arrays
np_weight_lb = np.array(weight_lb)
np_height_in = np.array(height_in)

# Print out the weight at index 50
print(weight_lb[50])

# Print out sub-array of np_height_in: index 100 up to and including index 110
print(np_height_in[100:111])


### --------------------------------------------------------
## 2D Numpy Arrays
# Create baseball, a list of lists
baseball = [[180, 78.4],
            [215, 102.7],
            [210, 98.5],
            [188, 75.2]]

# Import numpy
import numpy as np

# Create a 2D numpy array from baseball: np_baseball
np_baseball = np.array(baseball)


# Print out the type of np_baseball
print(type(np_baseball))

# Print out the shape of np_baseball
print(np_baseball.shape)


### --------------------------------------------------------
## Baseball data in 2D form
# baseball is available as a regular list of lists

# Import numpy package
import numpy as np

# Create a 2D numpy array from baseball: np_baseball
np_baseball = np.array(baseball)

# Print out the shape of np_baseball
print(np_baseball.shape)


### --------------------------------------------------------
## Subsetting 2D NumPy Arrays
# baseball is available as a regular list of lists
# Import numpy package
import numpy as np

# Create np_baseball (2 cols)
np_baseball = np.array(baseball)

# Print out the 50th row of np_baseball
print(np_baseball[49,:])

# Select the entire second column of np_baseball: np_weight_lb
np_weight_lb = np_baseball[:,1]

# Print out height of 124th player
print(np_baseball[123,0])


### --------------------------------------------------------
## 2D Arithmetic
# baseball is available as a regular list of lists
# updated is available as 2D numpy array

# Import numpy package
import numpy as np

# Create np_baseball (3 cols)
np_baseball = np.array(baseball)

# Print out addition of np_baseball and updated
print(np_baseball + updated)

# Create numpy array: conversion
conversion = np.array([0.0254, 0.453592, 1.0])

# Print out product of np_baseball and conversion
print(np_baseball * conversion)


### --------------------------------------------------------
## Numpy: Basic Statistics
### ----> Average versus median
# np_baseball is available
# Import numpy
import numpy as np

# Create np_height_in from np_baseball
np_height_in = np.array(np_baseball[:,0])

# Print out the mean of np_height_in
print(np.mean(np_height_in))
# Print out the median of np_height_in
print(np.median(np_height_in))


### --------------------------------------------------------
### Explore the baseball data
# np_baseball is available

# Import numpy
import numpy as np

# Print mean height (first column)
avg = np.mean(np_baseball[:,0])
print("Average: " + str(avg))

# Print median height. Replace 'None'
med = np.median(np_baseball[:,0])
print("Median: " + str(med))

# Print out the standard deviation on height. Replace 'None'
stddev = np.std(np_baseball[:,0])
print("Standard Deviation: " + str(stddev))

# Print out correlation between first and second column. Replace 'None'
corr = np.corrcoef(np_baseball[:,0], np_baseball[:,1])
print("Correlation: " + str(corr))



### --------------------------------------------------------
## ---> Blend it all together
# heights and positions are available as lists

# Import numpy
import numpy as np

# Convert positions and heights to numpy arrays: np_positions, np_heights
np_positions = np.array(positions)
np_heights = np.array(heights)


# Heights of the goalkeepers: gk_heights
gk_heights = np_heights[np_positions == 'GK']


# Heights of the other players: other_heights
other_heights = np_heights[np_positions != 'GK']

# Print out the median height of goalkeepers. Replace 'None'
print("Median height of goalkeepers: " + str(np.median(gk_heights)))

# Print out the median height of other players. Replace 'None'
print("Median height of other players: " + str(np.median(other_heights)))
