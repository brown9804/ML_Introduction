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
