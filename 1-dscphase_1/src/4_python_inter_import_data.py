#--  --  --  -- Intermediate Importing Data in Python:
# Used for Data Scientist Training Path 
#FYI it's a compilation of how to work
#with different commands.

### --------------------------------------------------------
# # ------>>>>> Importing flat files from the web: your turn!
# Import package
from urllib.request import urlretrieve
# Import pandas
import pandas as pd
# Assign url of file: url
url = 'https://s3.amazonaws.com/assets.datacamp.com/production/course_1606/datasets/winequality-red.csv'
# Save file locally
urlretrieve(url, 'winequality-red.csv')
# Read file into a DataFrame and print its head
df = pd.read_csv('winequality-red.csv', sep=';')
print(df.head())


### --------------------------------------------------------
# # ------>>>>> Opening and reading flat files from the web
# Import packages
import matplotlib.pyplot as plt
import pandas as pd
# Assign url of file: url
url = 'https://s3.amazonaws.com/assets.datacamp.com/production/course_1606/datasets/winequality-red.csv'
# Read file into a DataFrame: df
df = pd.read_csv(url, sep=';')
# Print the head of the DataFrame
print(df.head())
# Plot first column of df
pd.DataFrame.hist(df.ix[:, 0:1])
plt.xlabel('fixed acidity (g(tartaric acid)/dm$^3$)')
plt.ylabel('count')
plt.show()



### --------------------------------------------------------
# # ------>>>>> Importing non-flat files from the web
# Import package
import pandas as pd
# Assign url of file: url
url = 'http://s3.amazonaws.com/assets.datacamp.com/course/importing_data_into_r/latitude.xls'
# Read in all sheets of Excel file: xl
xls = pd.read_excel(url, sheetname=None)
# Print the sheetnames to the shell
print(xls.keys())
# Print the head of the first sheet (using its name, NOT its index)
print(xls['1700'].head())


### --------------------------------------------------------
# # ------>>>>> Performing HTTP requests in Python using urllib
# Import packages
from urllib.request import urlopen, Request
# Specify the url
url = "https://campus.datacamp.com/courses/1606/4135?ex=2"
# This packages the request: request
request = Request(url)
# Sends the request and catches the response: response
response = urlopen(request)
# Print the datatype of response
print(type(response))
# Be polite and close the response!
response.close()


### --------------------------------------------------------
# # ------>>>>> Printing HTTP request results in Python using urllib
# Import packages
from urllib.request import urlopen, Request
# Specify the url
url = "https://campus.datacamp.com/courses/1606/4135?ex=2"
# This packages the request
request = Request(url)
# Sends the request and catches the response: response
response = urlopen(request)
# Extract the response: html
html = response.read()
# Print the html
print(html)
# Be polite and close the response!
response.close()


### --------------------------------------------------------
# # ------>>>>> Performing HTTP requests in Python using requests
# Import package
import requests
# Specify the url: url
url = "http://www.datacamp.com/teach/documentation"
# Packages the request, send the request and catch the response: r
r = requests.get(url)
# Extract the response: text
text = r.text
# Print the html
print(text)


### --------------------------------------------------------
# # ------>>>>> Parsing HTML with BeautifulSoup
# Import packages
import requests
from bs4 import BeautifulSoup
# Specify url: url
url = 'https://www.python.org/~guido/'
# Package the request, send the request and catch the response: r
r = requests.get(url)
# Extracts the response as html: html_doc
html_doc = r.text
# Create a BeautifulSoup object from the HTML: soup
soup = BeautifulSoup(html_doc)
# Prettify the BeautifulSoup object: pretty_soup
pretty_soup = soup.prettify()
# Print the response
print(pretty_soup)


### --------------------------------------------------------
# # ------>>>>> Turning a webpage into data using BeautifulSoup: getting the text
# Import packages
import requests
from bs4 import BeautifulSoup
# Specify url: url
url = 'https://www.python.org/~guido/'
# Package the request, send the request and catch the response: r
r = requests.get(url)
# Extract the response as html: html_doc
html_doc = r.text
# Create a BeautifulSoup object from the HTML: soup
soup = BeautifulSoup(html_doc)
# Get the title of Guido's webpage: guido_title
guido_title = soup.title
# Print the title of Guido's webpage to the shell
print(guido_title)
# Get Guido's text: guido_text
guido_text = soup.get_text()
# Print Guido's text to the shell
print(guido_text)


### --------------------------------------------------------
# # ------>>>>> Turning a webpage into data using BeautifulSoup: getting the hyperlinks
# Import packages
import requests
from bs4 import BeautifulSoup
# Specify url
url = 'https://www.python.org/~guido/'
# Package the request, send the request and catch the response: r
r = requests.get(url)
# Extracts the response as html: html_doc
html_doc = r.text
# create a BeautifulSoup object from the HTML: soup
soup = BeautifulSoup(html_doc)
# Print the title of Guido's webpage
print(soup.title)
# Find all 'a' tags (which define hyperlinks): a_tags
a_tags = soup.find_all('a')
# Print the URLs to the shell
for link in a_tags:
    print(link.get('href'))


### --------------------------------------------------------
# # ------>>>>> Pop quiz: What exactly is a JSON?
# # Which of the following is NOT true of the JSON file format?
# JSONs consist of key-value pairs.
# JSONs are human-readable.
# The JSON file format arose out of a growing need for real-time server-to-browser communication.
# The function json.load() will load the JSON into Python as a list.
# The function json.load() will load the JSON into Python as a dictionary.
# R/ # The function json.load() will load the JSON into Python as a list.



### --------------------------------------------------------
# # ------>>>>> Loading and exploring a JSON
# Load JSON: json_data
with open("a_movie.json") as json_file:
    json_data = json.load(json_file)
# Print each key-value pair in json_data
for k in json_data.keys():
    print(k + ': ', json_data[k])


### --------------------------------------------------------
# # ------>>>>> Pop quiz: Exploring your JSON
# Load the JSON 'a_movie.json' into a variable, which will be a dictionary. 
# Do so by copying, pasting and executing the following code in the IPython Shell:
# import json
# with open("a_movie.json") as json_file:
#     json_data = json.load(json_file)
# Print the values corresponding to the keys 'Title' and 'Year' and 
# answer the following question about the movie that the JSON describes:
# Which of the following statements is true of the movie in question?
# R/ print(json_data) -> The title is 'The Social Network' and the year is 2010.



# ### --------------------------------------------------------
# # # ------>>>>> Pop quiz: What's an API?
# Which of the following statements about APIs is NOT true?
# An API is a set of protocols and routines for building and interacting with software applications.
# API is an acronym and is short for Application Program interface.
# It is common to pull data from APIs in the JSON file format.
# All APIs transmit data only in the JSON file format.
# An API is a bunch of code that allows two software programs to communicate with each other
# R/ ->>> # All APIs transmit data only in the JSON file format.



# ### --------------------------------------------------------
# # # ------>>>>> API requests
# Import requests package
import requests
# Assign URL to variable: url
url = 'http://www.omdbapi.com/?apikey=72bc447a&t=the+social+network'
# Package the request, send the request and catch the response: r
r = requests.get(url)
# Print the text of the response
print(r.text)


# ### --------------------------------------------------------
# # # ------>>>>> JSON–from the web to Python
# Import package
import requests
# Assign URL to variable: url
url = 'http://www.omdbapi.com/?apikey=72bc447a&t=social+network'
# Package the request, send the request and catch the response: r
r = requests.get(url)
# Decode the JSON data into a dictionary: json_data
json_data = r.json()
# Print each key-value pair in json_data
for k in json_data.keys():
    print(k + ': ', json_data[k])


# ### --------------------------------------------------------
# # # ------>>>>> Checking out the Wikipedia API
# Import package
import requests
# Assign URL to variable: url
url ='https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&exintro=&titles=pizza'
# Package the request, send the request and catch the response: r
r = requests.get(url)
# Decode the JSON data into a dictionary: json_data
json_data = r.json()
# Print the Wikipedia page extract
pizza_extract = json_data['query']['pages']['24768']['extract']
print(pizza_extract)


# ### --------------------------------------------------------
# # # ------>>>>> API Authentication
# Import package
import tweepy
# Store OAuth authentication credentials in relevant variables
access_token = "1092294848-aHN7DcRP9B4VMTQIhwqOYiB14YkW92fFO8k8EPy"
access_token_secret = "X4dHmhPfaksHcQ7SCbmZa2oYBBVSD2g8uIHXsp5CTaksx"
consumer_key = "nZ6EA0FxZ293SxGNg8g8aP0HM"
consumer_secret = "fJGEodwe3KiKUnsYJC3VRndj7jevVvXbK2D5EiJ2nehafRgA6i"
# Pass OAuth details to tweepy's OAuth handler
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)




# ### --------------------------------------------------------
# # # ------>>>>> Streaming tweets
# Initialize Stream listener
l = MyStreamListener()
# Create your Stream object with authentication
stream = tweepy.Stream(auth, l)
# Filter Twitter Streams to capture data by the keywords:
stream.filter(track=['clinton', 'trump', 'sanders', 'cruz'])


# ### --------------------------------------------------------
# # # ------>>>>> Load and explore your Twitter data
# Import package
import json
# String of path to file: tweets_data_path
tweets_data_path = 'tweets.txt'
# Initialize empty list to store tweets: tweets_data
tweets_data = []
# Open connection to file
tweets_file = open(tweets_data_path, "r")
# Read in tweets and store in list: tweets_data
for line in tweets_file:
    tweet = json.loads(line)
    tweets_data.append(tweet)
# Close connection to file
tweets_file.close()
# Print the keys of the first tweet dict
print(tweets_data[0].keys())


# ### --------------------------------------------------------
# # # ------>>>>> Twitter data to DataFrame
# Import package
import pandas as pd
# Build DataFrame of tweet texts and languages
df = pd.DataFrame(tweets_data, columns=['text', 'lang'])
# Print head of DataFrame
print(df.head())



# ### --------------------------------------------------------
# # # ------>>>>> A little bit of Twitter text analysis
# Initialize list to store tweet counts
[clinton, trump, sanders, cruz] = [0, 0, 0, 0]
# Iterate through df, counting the number of tweets in which
# each candidate is mentioned
for index, row in df.iterrows():
    clinton += word_in_text('clinton', row['text'])
    trump += word_in_text('trump', row['text'])
    sanders += word_in_text('sanders', row['text'])
    cruz += word_in_text('cruz', row['text'])



# ### --------------------------------------------------------
# # # ------>>>>> Plotting your Twitter data
# Import packages
import seaborn as sns
import matplotlib.pyplot as plt
# Set seaborn style
sns.set(color_codes=True)
# Create a list of labels:cd
cd = ['clinton', 'trump', 'sanders', 'cruz']
# Plot the bar chart
ax = sns.barplot(cd, [clinton, trump, sanders, cruz])
ax.set(ylabel="count")
plt.show()



# ### --------------------------------------------------------
# # # ------>>>>> 


# ### --------------------------------------------------------
# # # ------>>>>> 


# ### --------------------------------------------------------
# # # ------>>>>> 