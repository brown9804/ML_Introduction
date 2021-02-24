# #--  --  --  -- Building and Distributing Packages with Conda
# # Used for Data Scientist Training Path 
# #FYI it's a compilation of how to work
# #with different commands.

# ### --------------------------------------------------------
# # # ------>>>> What is an Anaconda Project? 
# In the Conda Essentials 
# course you learned how use 
# the Conda package manager to 
# create and share reproducible 
# environments for data science 
# development. 

# In this chapter you'll create 
# an Anaconda Project, which is 
# a data science asset that 
# specifies package installs, 
# file downloads, and 
# executable commands. Anaconda 
# projects can be used to run 
# Jupyter notebooks, Bokeh 
# server apps, REST APIs, and 
# command line tools on 
# Windows, Mac OSX, and Linux 
# platforms making deployment 
# easy. 

# Anaconda projects are shared 
# amongst data scientists as 
# compressed directories 
# containing the Conda 
# environment specification, 
# URLs for downloadable files, 
# and source code for commands. 
# Collaboration can be achieved 
# through text file revision 
# control tools, like git. 

# In the following exercises 
# you'll learn how to create, 
# run, and share Anaconda 
# Projects. 

# Which of the following CANNOT 
# be achieved with an Anaconda 
# Project?
# # Develop a project on your workstation and deploy it to a production web server.
# Automatically install the correct version of each required package before running a command.
# Provide an on-line editing environment for multiple users to develop code.
# Define datasets to be downloaded before running commands.
# R/ Provide an on-line editing environment for multiple users to develop code.


# ### --------------------------------------------------------
# # # ------>>>> Install Anaconda-Project
conda install anaconda-project -y


# ### --------------------------------------------------------
# # # ------>>>>Prepare and run a project command
anaconda-project list-packages
anaconda-project prepare
anaconda-project list-commands
anaconda-project run search-names Belinda


# # ### --------------------------------------------------------
# # # # ------>>>>Anaconda Project specification file
# The core 
# of an Anaconda Project is a 
# YAML file containing a 
# specification of the conda 
# packages, commands, and 
# downloads that make up the 
# Project. 

# The YAML file is called 
# anaconda-project.yml and each 
# separate project you create 
# will be in it's own 
# subdirectory containing a 
# distinct anaconda-project.yml 
# file for that project. 

# In the terminal you will see 
# that we have navigated to the 
# babynames directory. Use ls 
# to inspect the contents of 
# this directory. Further, 
# inspect the anaconda-
# project.yml file. You should 
# see YAML tags for packages, 
# commands, and downloads. 

# Choose the correct command 
# below that was used when you 
# executed anaconda-project run 
# search-names <NAME> in the 
# previous exercise. You can 
# use tools like nano, vim, 
# emacs, cat, less, or more to 
# read the file
# R/--> python main.py



# # ### --------------------------------------------------------
# # # # ------>>>>Initialize a new project
mkdir mortgage_rates
cd mortgage_rates
anaconda-project init
nano anaconda-project.yml
# adding -> description: Forecast 30-year mortgage rates in the US


# # # ### --------------------------------------------------------
# # # # # ------>>>>Anaconda Project commands As 
# you have seen Anaconda 
# Projects provide reproducible 
# execution of data science 
# assets. The babynames project 
# defined a command-line-
# interface (CLI) command to 
# analyze yearly trends. 

# When defining a command line 
# tool in Anaconda Project the 
# type unix or windows should 
# be used. Typically both are 
# defined, where unix is the 
# full command as run in Bash 
# and windows is the full 
# command as run in the Windows 
# Shell. 

# Anaconda Projects can support 
# four types of commands: 

# Unix commands: shell-based 
# commands that run on Mac or 
# Unix systems Windows 
# commands: Windows shell 
# commands Bokeh App: Run bokeh-
# server with a given Python 
# script Jupyter Notebook: 
# Launch Jupyter Notebook with 
# the specified notebook file 
# For both Unix and Windows 
# commands any arbitrary 
# command can be run. These 
# could be OS-specific tools or 
# Python scripts provided with 
# the Project. The Conda 
# environment defined in 
# anaconda-project.yml is 
# created and activated 
# automatically when running a 
# command. 

# Commands are added to 
# projects using the anaconda-
# project add-command command. 
# Projects can have any number 
# of commands defined. 

# Which of the following tasks 
# are not supported by Anaconda 
# Project commands?
# Launch a rich web dashboard built with Bokeh
# Start a light-weight REST API built with Python
# Launch a graphical user interface (GUI) tool
# R/ None of the above


# # # ### --------------------------------------------------------
# # # # # ------>>>>Add packages and commands
anaconda-project add-packages pandas
anaconda-project add-download MORTGAGE_RATES https://goo.gl/jpbAsR
nano forescast.py #-> change MORTGAGE_RATES = os.environ["MORTGAGE_RATES"]
anaconda-project add-command --type unix forecast "python forecast.py"
anaconda-project run forecast


# # # ### --------------------------------------------------------
# # # # # ------>>>>Locking package versions
anaconda-project lock


# # # ### --------------------------------------------------------
# # # # # ------>>>>Sharing your project
anaconda-project archive ../mortgage_rates.zip
anaconda login --username datacamp-student --password datacamp1
anaconda upload ../mortgage_rates.zip -t


# # # ### --------------------------------------------------------
# # # # # ------>>>>Python modules and packages 
# You are familiar with Python 
# scripts that can be executed 
# by running python <script-
# file>.py. Each of these files 
# has one or more import 
# statements to re-use Python 
# source code written by other 
# developers. These re-usable 
# Python source files are 
# referred to as modules. They 
# are .py Python source files, 
# like the scripts you have 
# been writing, that are stored 
# in a location where the 
# import statement can find 
# them. 

# There is no overtly 
# recognizable difference 
# between Python modules and 
# scripts, except that 
# developers of the former 
# write them with the intention 
# that their contents are 
# imported, rather than 
# executed by users. 

# In this chapter you'll learn 
# how to turn your Python 
# scripts into importable 
# modules, collect those 
# modules into packages and 
# create Conda packages that 
# can be easily installed by 
# other users. 

# Which of the following is 
# true about Python modules?
# R/ Python modules must be installed in order to use them.


# # # ### --------------------------------------------------------
# # # # # ------>>>> Importing a module
python pi.py compute_pi(1000)
# R/ ---> 3.14


# # # ### --------------------------------------------------------
# # # # # ------>>>> Modules and __name__
# The import statement was faster.


# # # ### --------------------------------------------------------
# # # # # ------>>>>Python package directory So 
# far you have seen that any 
# .py Python source file can be 
# used as a module to import 
# code. A Python package is a 
# collection of separate 
# modules collected under a 
# single name that share 
# metadata, such as 
# documentation, licensing, and 
# version numbering. 

# Here's an example directory 
# structure for my_package. 
# Note the repeated use of 
# my_package directories. The 
# individual modules are stored 
# in the inner my_package 
# directory. 

# my_package/ LICENSE README 
# setup.py my_package/ 
# __init__.py module1.py 
# module2.py ... Once this 
# package has been installed 
# you can import code from the 
# modules using the standard 
# import idioms. 

# Over the next several 
# exercises you're going to re-
# use much of the same code 
# that was present in the 
# mortgage_rates Anaconda 
# Project to prepare a Python 
# package that can be installed 
# and imported in new Python 
# projects. 

# The setup.py script file 
# coordinates installing the 
# package into your Python 
# distribution. 

# Why is it important to 
# package re-usable Python code?
# To keep a consistent version history of changes.
# To make module installation and management easier for others.
# To clearly define importable API components.
# R/ All of the above.



# # # ### --------------------------------------------------------
# # # # # # ------>>>>Importing a package A Python 
# package can have any number 
# of directories and module 
# source files. 

# The directory 
# mortgage_forecasts has been 
# prepared for you in your home 
# directory. It contains a 
# subdirectory of the same name 
# and two Python source code 
# files. 

# mortgage_forecasts/ 
# mortgage_forecasts/ models.py 
# utils.py models.py defines a 
# new class to fit and predict 
# 30-year mortgage rates in the 
# US. 

# utils.py defines functions to 
# read data and compute 
# statistical quantities. 

# Your working directory has 
# been set to 
# /home/repl/mortgage_forecasts. 
# Without changing directories, 
# which import statement will 
# provide access to the 
# MortgageRateModel class 
# defined in models.py? 

# You can use the terminal to 
# test import statements. A 
# conda environment has been 
# activated with the dependent 
# packages.
# R/ from mortgage_forecasts.models import MortgageRateModel



# # # ### --------------------------------------------------------
# # # # # ------>>>> The __init__.py file
echo "'''Predictive modeling of 30-year mortgage rates in the US.'''" > /home/repl/mortgage_forecasts/mortgage_forecasts/__init__.py
echo "from .models import MortgageRateModel" >> /home/repl/mortgage_forecasts/mortgage_forecasts/__init__.py; echo "from .utils import read_data" >> /home/repl/mortgage_forecasts/mortgage_forecasts/__init__.py


# # # ### --------------------------------------------------------
# # # # # ------>>>>Create the installer script
# Open any of the available editors to edit the file instead of using the following command
sed -i -e 's/name =.*/name = "mortgage_forecasts",/' /home/repl/mortgage_forecasts/setup.py; sed -i -e 's/description =.*/description = "30 year mortgage rate models",/' /home/repl/mortgage_forecasts/setup.py; sed -i -e 's/author =.*/author = "",/' /home/repl/mortgage_forecasts/setup.py
sed -i -e 's/setup(/setup( packages=find_packages(),/' /home/repl/mortgage_forecasts/setup.py

# # # # ### --------------------------------------------------------
# # # # # # ------>>>> Licensing Since our goal is 
# to share our code with others 
# we need to be aware of 
# copyright laws and the legal 
# rights we wish to retain 
# about how that software can 
# be used. Copyright 
# protections are guaranteed to 
# the person who owns the 
# software. When someone else 
# downloads and uses the 
# program we built we would not 
# want to transfer ownership to 
# them, thereby forfeiting our 
# rights. 

# Instead, we wish license 
# usage of the program under 
# certain restrictions. There 
# are many kinds of software 
# licenses and choosing a 
# license is beyond the scope 
# of this course. 

# For the mortgage_forecasts 
# package we want to share our 
# code with as few restrictions 
# as necessary. The only 
# important restriction is that 
# we would like to be 
# acknowledged as the original 
# author. For this we'll choose 
# the MIT license, which is 
# widely used in open-source 
# software packages. 

# The full text of the license 
# has been placed in the 
# mortgage_forecasts/LICENSE 
# file and it must remain in 
# the package directory for the 
# license to be valid and 
# enforceable. 

# Further, license="MIT" has 
# been added in setup.py. 

# which statement below is 
# INCORRECT?
# R/The MIT license does not allow commercial use.


# # # ### --------------------------------------------------------
# # # # # ------>>>> Version number
echo "__version__ = '0.1'" >> /home/repl/mortgage_forecasts/mortgage_forecasts/__init__.py

# # # ### --------------------------------------------------------
# # # # # ------>>>>Install the package
python setup.py install
cd /home/repl/practice
# In the Python shell import MortgageRateModel and read_data from the mortgage_forecasts package and call:

# read_data() with the argument mortgage_rates.csv -> df
# MortgageRateModel() with the argument df -> model
# model.forecast() with the argument 'January 2019'
# You can continue to inspect the help() of the package, functions, and classes.

# Exit the interpreter when you're done.




# # # ### --------------------------------------------------------
# # # # # ------>>>>Conda Packages In the last 
# chapter you created a Python 
# package and successfully 
# installed it. However, you 
# needed to have 1) downloaded 
# the source code, and 2) 
# created a Conda Environment 
# with the dependent packages. 

# Further, when using 
# setuptools to install 
# packages there are no 
# uninstall or update commands. 
# That means you would have to 
# manually remove the installed 
# files if you want to install 
# a newer version of the 
# package. As you saw in the 
# Conda Essentials course, 
# Conda packages solve each of 
# these issues, but you might 
# use pip and virtualenv as 
# well. 

# In this chapter you'll create 
# a Conda Recipe for the 
# mortgage_forecasts package to 
# define the dependent Conda 
# packages. The Conda recipe is 
# specified in a file called 
# meta.yaml. 

# You'll then build the package 
# archive and upload it to 
# Anaconda Cloud. Further, 
# Conda packages are not 
# limited to Python packages. A 
# package written in any 
# programming language, or a 
# collection of files, can 
# become a Conda package. 

# Which statement below is 
# INCORRECT?
# Having a setup.py file is enough to build a Conda package.

# # # ### --------------------------------------------------------
# # # # # ------>>>> Install Conda Build
conda install conda-build -y

# # # # ### --------------------------------------------------------
# # # # # # ------>>>> The Conda recipe meta.yaml
# The mortgage_forecasts 
# package you wrote in the last 
# chapter has been provided in 
# your home directory along 
# with a template meta.yaml 
# Conda recipe. There are 5 
# sections of this file: 

# package defines the package 
# name and version source 
# provides the relative path (
# or Github repository) to the 
# package source files build 
# defines the command to 
# install the package 
# requirements specify the 
# conda packages required to 
# build and run the package 
# about provides other 
# important metadata like the 
# license and description 
# Inspect the meta.yaml file 
# with nano, vim, emacs, cat, 
# less or more. 

# You'll see at the top of the 
# file {% set setup_py = 
# load_setup_py_data() %}. When 
# the package is built metadata 
# like the version number and 
# license will be read directly 
# from the setup.py file in the 
# source path. 

# Read the meta.yaml 
# documentation for more 
# details. 

# Why are there build: and run: 
# sections in requirements:?
# R/Build packages are those required by setup.py.


# # # ### --------------------------------------------------------
# # # # # # ------>>>>Conda package dependencies
# In the meta.yaml file dependent packages and versions are defined using 
# comparison operators, such as <, <=, >, >=. Multiple version conditions
#  are separated by commas. The glob * means that the preceding characters must match exactly.

# Here's an example for a new package called my_package.

# requirements:
#     run:
#         - python
#         - scipy
#         - numpy 1.11*
#         - matplotlib >=1.5, <2.0
# Which of the following statements below is INCORRECT?
# R/NumPy version 1.13 is compatible with my_package.


# # # ### --------------------------------------------------------
# # # # # ------>>>>Dependent package versions
sed -i -e 's/^    run:/    run:\n         - python >=2.7\n        - pandas >=0.20/n        - statsmodels\n        - scipy\n' /home/repl/mortgage_forecasts/meta.yaml
sed -i -e 's/^    build:/    build:\n        - python\n        - setuptools/' /home/repl/mortgage_forecasts/meta.yaml


# # # ### --------------------------------------------------------
# # # # # ------>>>>Build the Conda Package
conda build mortgage_forecasts
conda search --use-local --info mortgage_forecasts


# # # ### --------------------------------------------------------
# # # # # ------>>>Install the conda package
conda create -n conda-models python=3
conda activate models 
conda install mortgage_forecasts pandas=0.19 --use-local
conda install mortgage_forecasts pandas=0.22 --use-local


# # # ### --------------------------------------------------------
# # # # # ------>>>Python versions and architectures
# Edit meta.yaml to add the following line to the build: tag and before the script: tag.

#     noarch: python
#     number: 1
conda build .


# # # ### --------------------------------------------------------
# # # # # ------>>>Upload the package
anaconda upload noarch/mortgage_forecasts-0.1-py_1.tar.bz2 --username datacamp-student --password datacamp1
