  #--  --  --  -- Statistical Thinking in Python (Part 1):
# Used for Data Scientist Training Path 
#FYI it's a compilation of how to work
#with different commands.

### --------------------------------------------------------
# ------>>>>>    Tukey's comments on EDA 
# (Exploratory Data Analysis )
# Even though you probably have not read Tukey's book, I 
# suspect you already have a good idea about his viewpoint 
# from the video introducing you to exploratory data analysis.
#  Which of the following quotes is not directly from Tukey?
# R/ Often times EDA is too time consuming, so it is better to 
# jump right in and do your hypothesis tests.


### --------------------------------------------------------
# ------>>>>> Advantages of graphical EDA  
# Which of the following is not true of graphical EDA?
# R/ A nice looking plot is always the end goal of a statistical analysis.


### --------------------------------------------------------
# ------>>>>> Plotting a histogram of iris data
# Import plotting modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Set default Seaborn style
sns.set()
# Plot histogram of versicolor petal lengths
plt.hist(versicolor_petal_length)
# Show histogram
plt.show()


### --------------------------------------------------------
# ------>>>>>  Axis labels!
# Import plotting modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Plot histogram of versicolor petal lengths
plt.hist(versicolor_petal_length)
# Set default Seaborn style
sns.set()
# Label axes
plt.xlabel('petal length (cm)')
plt.ylabel('count')
# Show histogram
plt.show()


### --------------------------------------------------------
####-----> Adjusting the number of bins in a histogram
# Import numpy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Compute number of data points: n_data
n_data = len(versicolor_petal_length)
# Number of bins is the square root of number of data points: n_bins
n_bins = np.sqrt(n_data)
# Convert number of bins to integer: n_bins
n_bins = int(n_bins)
# Plot the histogram
plt.hist(versicolor_petal_length, bins=n_bins)
# Label axes
plt.xlabel('petal length (cm)')
plt.ylabel('count')
# Show histogram
plt.show()


### --------------------------------------------------------
#### ----> Bee swarm plot
## Import 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Create bee swarm plot with Seaborn's default settings
sns.swarmplot(x='species', y='petal length (cm)', data=df)
# Label the axes
plt.xlabel('species')
plt.ylabel('petal length (cm)')
# Show the plot
plt.show()


### --------------------------------------------------------
#### ----> Interpreting a bee swarm plot
# Which of the following conclusions could you draw from 
# the bee swarm plot of iris petal lengths you generated 
# in the previous exercise? For your convenience, the bee 
# swarm plot is regenerated and shown to the right.
## R/ I. virginica petals tend to be the longest, and I. 
# setosa petals tend to be the shortest of the three species.


### --------------------------------------------------------
###----> Computing the ECDF
import numpy as np
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n
    return x, y


### --------------------------------------------------------
###----> Plotting the ECDF
import numpy as np
import matplotlib.pyplot as plt
# Compute ECDF for versicolor data: x_vers, y_vers
x_vers, y_vers = ecdf(versicolor_petal_length)
# Generate plot
plt.plot(x_vers, y_vers, marker = '.', linestyle = 'none')
# Make the margins nice
plt.margins(0.02)
# Label the axes
plt.xlabel('length')
plt.ylabel('ECDF')
# Display the plot
plt.show()


### --------------------------------------------------------
###----> Comparison of ECDFs
import numpy as np
import matplotlib.pyplot as plt
# Compute ECDFs
x_set, y_set = ecdf(setosa_petal_length)
x_vers, y_vers = ecdf(versicolor_petal_length)
x_virg, y_virg = ecdf(virginica_petal_length)
# Plot all ECDFs on the same plot
plt.plot(x_set, y_set, marker = '.', linestyle = 'none')
plt.plot(x_vers, y_vers, marker = '.', linestyle = 'none')
plt.plot(x_virg, y_virg, marker = '.', linestyle = 'none')
# Make nice margins
plt.margins(0.02)
# Annotate the plot
plt.legend(('setosa', 'versicolor', 'virginica'), loc='lower right')
plt.xlabel('petal length (cm)')
plt.ylabel('ECDF')
# Display the plot
plt.show()


### --------------------------------------------------------
###----> Means and medians
# Which one of the following statements is true about means and medians?
# R/ An outlier can significantly affect the value of the mean, but not the median.


### --------------------------------------------------------
###------> Computing means
import numpy as np
# Compute the mean: mean_length_vers
mean_length_vers = np.mean(versicolor_petal_length)
# Print the result with some nice formatting
print('I. versicolor:', mean_length_vers, 'cm')


### --------------------------------------------------------
###------> Computing percentiles
import numpy as np
import seaborn as sns
# Specify array of percentiles: percentiles
percentiles = np.array([2.5, 25, 50, 75, 97.5])
# Compute percentiles: ptiles_vers
ptiles_vers = np.percentile(versicolor_petal_length, percentiles)
# Print the result
print(ptiles_vers)


### --------------------------------------------------------
###------> Comparing percentiles to ECDF
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Plot the ECDF
plt.plot(x_vers, y_vers, '.')
plt.margins(0.02)
plt.xlabel('petal length (cm)')
plt.ylabel('ECDF')
# Overlay percentiles as red diamonds.
plt.plot(ptiles_vers, percentiles / 100, marker='D', color='red',
             linestyle='none')
# Show the plot
plt.show()


### --------------------------------------------------------
###------> Box-and-whisker plot
# Create box plot with Seaborn's default settings
sns.boxplot(x='species', y='petal length (cm)', data=df)
# Label the axes
plt.xlabel('species')
plt.ylabel('petal length (cm)')
# Show the plot
plt.show()


### --------------------------------------------------------
###------> Computing the variance
# Array of differences to mean: differences
differences = np.array(versicolor_petal_length -
                       np.mean(versicolor_petal_length))
# Square the differences: diff_sq
diff_sq = differences ** 2
# Compute the mean square difference: variance_explicit
variance_explicit = np.mean(diff_sq)
# Compute the variance using NumPy: variance_np
variance_np = np.var(versicolor_petal_length)
# Print the results
print(variance_explicit, variance_np)


### --------------------------------------------------------
###------> The standard deviation and the variance
# Compute the variance: variance
variance = np.var(versicolor_petal_length)
# Print the square root of the variance
print(np.sqrt(variance))
# Print the standard deviation
print(np.std(versicolor_petal_length))


#### Covariance and the Pearson correlation coefficient
### --------------------------------------------------------
###------> Scatter plots
# Make a scatter plot
plt.plot(versicolor_petal_length, versicolor_petal_width,
             marker='.', linestyle='none')
# Set margins
plt.margins(0.02)
# Label the axes
plt.xlabel('versicolor petal length')
plt.ylabel('versicolor petal width')
# Show the result
plt.show()


### --------------------------------------------------------
###------> Variance and covariance by looking
# Consider four scatter plots of x-y data, appearing to the 
# right. Which has, respectively,
# the highest variance in the variable ,
# the highest covariance,
# negative covariance?
# R/ d, c, b


### --------------------------------------------------------
###------> Computing the covariance
# Compute the covariance matrix: covariance_matrix
covariance_matrix = np.cov(versicolor_petal_length, versicolor_petal_width)
# Print covariance matrix
print(covariance_matrix)
# Extract covariance of length and width of petals: petal_cov
petal_cov = covariance_matrix[0, 1]
# Print the length/width covariance
print(petal_cov)



### --------------------------------------------------------
###------> Computing the Pearson correlation coefficient
def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x, y)
    # Return entry [0,1]
    return corr_mat[0, 1]
# Compute Pearson correlation coefficient for I. versicolor: r
r = pearson_r(versicolor_petal_length, versicolor_petal_width)
# Print the result
print(r)


### --------------------------------------------------------
###------> What is the goal of statistical inference?
# Why do we do statistical inference?
# R/ To draw probabilistic conclusions about what we might expect if we collected the same data again.
# To draw actionable conclusions from data.
# To draw more general conclusions from relatively few data or observations.
# All of these. -----> answer


### --------------------------------------------------------
###------> Why do we use the language of probability?
# Which of the following is not a reason why we use probabilistic 
# language in statistical inference?
# R/ Probabilistic language is not very precise.


### --------------------------------------------------------
###------> Generating random numbers using the np.random module
import numpy as np
import matplotlib.pyplot as plt
# Seed the random number generator
np.random.seed(42)
# Initialize random numbers: random_numbers
random_numbers = np.empty(100000)
# Generate random numbers by looping over range(100000)
for i in range(100000):
    random_numbers[i] = np.random.random()
    print(random_numbers[i])
# Plot a histogram
plt.hist(random_numbers)
# Show the plot
plt.show()


### --------------------------------------------------------
###------>The np.random module and Bernoulli trials
def perform_bernoulli_trials(n, p):
    """Perform n Bernoulli trials with success probability p
    and return number of successes."""
    # Initialize number of successes: n_success
    n_success = 0
    # Perform trials
    for i in range(n):
        # Choose random number between zero and one: random_number
        random_number = np.random.random()
        # If less than p, it's a success so add one to n_success
        if random_number < p:
            n_success += 1
    return n_success
print(perform_bernoulli_trials(10000, 0.65))


### --------------------------------------------------------
###------>How many defaults might we expect?
def perform_bernoulli_trials(n, p):
    """Perform n Bernoulli trials with success probability p
    and return number of successes."""
    # Initialize number of successes: n_success
    n_success = 0
    # Perform trials
    for i in range(n):
        # Choose random number between zero and one: random_number
        random_number = np.random.random()
        # If less than p, it's a success so add one to n_success
        if random_number < p:
            n_success += 1
    return n_success
# Seed random number generator
np.random.seed(42)
# Initialize the number of defaults: n_defaults
n_defaults = np.empty(1000)
# Compute the number of defaults
for i in range(1000):
    n_defaults[i] = perform_bernoulli_trials(100, 0.05)
# Plot the histogram with default number of bins; label your axes
plt.hist(n_defaults, normed=True)
plt.xlabel('number of defaults out of 100 loans')
plt.ylabel('probability')
# Show the plot
plt.show()


### --------------------------------------------------------
###------>Will the bank fail?
# Compute ECDF: x, y
x, y = ecdf(n_defaults)
# Plot the ECDF with labeled axes
plt.plot(x, y, marker='.', linestyle='none')
plt.xlabel('x')
plt.ylabel('y')
# Show the plot
plt.show()
# Compute the number of 100-loan simulations with 10 or more defaults: n_lose_money
n_lose_money = np.sum(n_defaults >= 10)
# Compute and print probability of losing money
print('Probability of losing money =', n_lose_money / len(n_defaults))



### --------------------------------------------------------
###------> Sampling out of the Binomial distribution
# Take 10,000 samples out of the binomial distribution: n_defaults
n_defaults = np.random.binomial(n=100, p=0.05, size=10000)
# Compute CDF: x, y
x, y = ecdf(n_defaults)
# Plot the CDF with axis labels
plt.plot(x, y, marker='.', linestyle='none')
plt.xlabel('Defaults out of 100')
plt.ylabel('CDF')
# Show the plot
plt.show()


### --------------------------------------------------------
###------> Plotting the Binomial PMF
# Compute bin edges: bins
bins = np.arange(min(n_defaults), max(n_defaults) + 1.5) - 0.5
# Generate histogram
plt.hist(n_defaults, normed=True, bins=bins)
# Set margins
plt.margins(0.02)
# Label axes
plt.xlabel('x')
plt.ylabel('y')
# Show the plot
plt.show()

### --------------------------------------------------------
###------> Relationship between Binomial and Poisson distributions
# Draw 10,000 samples out of Poisson distribution: samples_poisson
samples_poisson = np.random.poisson(10, size=10000)
# Print the mean and standard deviation
print('Poisson:     ', np.mean(samples_poisson),
                       np.std(samples_poisson))
# Specify values of n and p to consider for Binomial: n, p
n = [20, 100, 1000]
p = [0.5, 0.1, 0.01]
# Draw 10,000 samples for each n,p pair: samples_binomial
for i in range(3):
    samples_binomial = np.random.binomial(n[i], p[i], 10000)
    # Print results
    print('n =', n[i], 'Binom:', np.mean(samples_binomial),
                                 np.std(samples_binomial))


### --------------------------------------------------------
###------> How many no-hitters in a season?
# In baseball, a no-hitter is a game in which a pitcher 
# does not allow the other team to get a hit. This is a 
# rare event, and since the beginning of the so-called 
# modern era of baseball (starting in 1901), there have only 
# been 251 of them through the 2015 season in over 200,000 games. 
# The ECDF of the number of no-hitters in a season is shown to 
# the right. Which probability distribution would be appropriate 
# to describe the number of no-hitters we would expect in a given season?
# Note: The no-hitter data set was scraped and calculated from the 
# data sets available at retrosheet.org (license).
# R/ Both Binomial and Poisson, though Poisson is easier to model and compute.


### --------------------------------------------------------
###------> Was 2015 anomalous?
# Draw 10,000 samples out of Poisson distribution: n_nohitters
n_nohitters = np.random.poisson((251 / 115), size=10000)
# Compute number of samples that are seven or greater: n_large
n_large = np.sum(n_nohitters >= 7)
# Compute probability of getting seven or more: p_large
p_large = n_large / 10000
# Print the result
print('Probability of seven or more no-hitters:', p_large)


### --------------------------------------------------------
###------> Interpreting PDFs - probability density function 
# Consider the PDF shown to the right (it may take a second to load!).
#  Which of the following is true?
# R/  is more likely to be less than 10 than to be greater than 10.


### --------------------------------------------------------
###------> Interpreting CDFs ---- cumulative distribution function 
# At right is the CDF corresponding to the PDF you considered in the
#  last exercise. Using the CDF, what is the probability that  is greater than 10?
# R/ 0.25


### --------------------------------------------------------
###------> The Normal PDF
# Draw 100000 samples from Normal distribution with stds of interest: samples_std1, samples_std3,
#   samples_std10
samples_std1 = np.random.normal(20, 1, size=100000)
samples_std3 = np.random.normal(20, 3, size=100000)
samples_std10 = np.random.normal(20, 10, size=100000)
# Make histograms
plt.hist(samples_std1, bins=100, normed=True, histtype='step')
plt.hist(samples_std3, bins=100, normed=True, histtype='step')
plt.hist(samples_std10, bins=100, normed=True, histtype='step')
# Make a legend, set limits and show plot
plt.legend(('std = 1', 'std = 3', 'std = 10'))
plt.ylim(-0.01, 0.42)
plt.show()



### --------------------------------------------------------
###------>The Normal CDF
# Generate CDFs
x_std1, y_std1 = ecdf(samples_std1)
x_std3, y_std3 = ecdf(samples_std3)
x_std10, y_std10 = ecdf(samples_std10)
# Plot CDFs
plt.plot(x_std1, y_std1, marker='.', linestyle='none')
plt.plot(x_std3, y_std3, marker='.', linestyle='none')
plt.plot(x_std10, y_std10, marker='.', linestyle='none')
# Make 2% margin
plt.margins(0.02)
# Make a legend and show the plot
plt.legend(('std = 1', 'std = 3', 'std = 10'), loc='lower right')
plt.show()


### --------------------------------------------------------
###------> Gauss and the 10 Deutschmark banknote
# What are the mean and standard deviation, respectively,
# of the Normal distribution that was on the 10 Deutschmark banknote, 
# shown to the right?
# R/ mean = 3, std = 1


### --------------------------------------------------------
###------> Are the Belmont Stakes results Normally distributed?
# Compute mean and standard deviation: mu, sigma
mu = np.mean(belmont_no_outliers)
sigma = np.std(belmont_no_outliers)
# Sample out of a normal distribution with this mu and sigma: samples
samples = np.random.normal(mu, sigma, 10000)
# Get the CDF of the samples and of the data
x_theor, y_theor = ecdf(samples)
x, y = ecdf(belmont_no_outliers)
# Plot the CDFs and show the plot
plt.plot(x_theor, y_theor)
plt.plot(x, y, marker='.', linestyle='none')
plt.margins(0.02)
plt.xlabel('Belmont winning time (sec.)')
plt.ylabel('CDF')
plt.show()

### --------------------------------------------------------
###------> What are the chances of a horse matching or
#  beating Secretariat's record?
# Take a million samples out of the Normal distribution: samples
samples = np.random.normal(mu, sigma, 1000000)
# Compute the fraction that are faster than 144 seconds: prob
prob = len(samples[np.where(samples <= 144)]) / len(samples)
# Print the result
print('Probability of besting Secretariat:', prob)


### --------------------------------------------------------
###------> Matching a story and a distribution
# How might we expect the time between Major League no-hitters to
# be distributed? Be careful here: a few exercises ago, we considered
# the probability distribution for the number of no-hitters in a season.
# Now, we are looking at the probability distribution of the time 
# between no hitters.
# R/ Exponential


### --------------------------------------------------------
###------>Waiting for the next Secretariat
# Unfortunately, Justin was not alive when Secretariat ran
#  the Belmont in 1973. Do you think he will get to see a 
# performance like that? To answer this, you are interested in 
# how many years you would expect to wait until you see another 
# performance like Secretariat's. How is the waiting time until 
# the next performance as good or better than Secretariat's distributed? 
# Choose the best answer.
# R/ Exponential: A horse as fast as Secretariat is a rare event, which can be modeled as
#  a Poisson process, and the waiting time between arrivals of a Poisson
#  process is Exponentially distributed.



### --------------------------------------------------------
###------>If you have a story, you can simulate it!
def successive_poisson(tau1, tau2, size=1):
    """Compute time for arrival of 2 successive Poisson processes."""
    # Draw samples out of first exponential distribution: t1
    t1 = np.random.exponential(tau1, size)
    # Draw samples out of second exponential distribution: t2
    t2 = np.random.exponential(tau2, size)
    return t1 + t2


### --------------------------------------------------------
###------> Distribution of no-hitters and cycles
# Draw samples of waiting times: waiting_times
waiting_times = np.array(successive_poisson(764, 715, 100000))
# Make the histogram
plt.hist(waiting_times, bins=100, normed=True, histtype='step')
# Label axes
plt.xlabel('x')
plt.ylabel('y')
# Show the plot
plt.show()




# probability mass function (PMF)
