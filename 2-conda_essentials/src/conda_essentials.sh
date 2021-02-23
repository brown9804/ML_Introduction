# #--  --  --  -- Conda Essentials
# # Used for Data Scientist Training Path 
# #FYI it's a compilation of how to work
# #with different commands.

# ### --------------------------------------------------------
# # # ------>>>>What are packages and why are 
# they needed? Conda packages 
# are files containing a bundle 
# of resources: usually 
# libraries and executables, 
# but not always. In principle, 
# Conda packages can include 
# data, images, notebooks, or 
# other assets. The command-
# line tool conda is used to 
# install, remove and examine 
# packages; other tools such as 
# the GUI Anaconda Navigator 
# also expose the same 
# capabilities. This course 
# focuses on the conda tool 
# itself (you'll see use cases 
# other than package management 
# in later chapters). 

# Conda packages are most 
# widely used with Python, but 
# that's not all. Nothing about 
# the Conda package format or 
# the conda tool itself assumes 
# any specific programming 
# language. Conda packages can 
# also be used for bundling 
# libraries in other languages (
# like R, Scala, Julia, etc.) 
# or simply for distributing 
# pure binary executables 
# generated from any 
# programming language. 

# One of the powerful aspects 
# of conda—both the tool and 
# the package format—is that 
# dependencies are taken care 
# of. That is, when you install 
# any Conda package, any other 
# packages needed get installed 
# automatically. Tracking and 
# determining software 
# dependencies is a hard 
# problem that package managers 
# like Conda are designed to 
# solve. 

# A Conda package, then, is a 
# file containing all files 
# needed to make a given 
# program execute correctly on 
# a given system. Moreover, a 
# Conda package can contain 
# binary artifacts specific to 
# a particular platform or 
# operating system. Most 
# packages (and their 
# dependencies) are available 
# for Windows (win-32 or win-
# 64), for OSX (osx-64), and 
# for Linux (linux-32 or linux-
# 64). A small number of Conda 
# packages are available for 
# more specialized platforms (
# e.g., Raspberry Pi 2 or 
# POWER8 LE). As a user, you do 
# not need to specify the 
# platform since Conda will 
# simply choose the Conda 
# package appropriate for the 
# platform you are using. 

# ### --------------------------------------------------------
# # # ------>>>>What feature is NOT true of 
# Conda packages?
# R/Each package contains binary artifacts (executables) for all supported platforms.


# ### --------------------------------------------------------
# # # ------>>>> What version of conda do I have?
conda --version

# ### --------------------------------------------------------
# # # ------>>>>Install a conda package (I)
# Just as you can get help on conda as a whole, 
# you can get help on commands within it. You will 
# often use the command conda install; you can look
#  at the corresponding help documentation using the 
#  terminal window. That is, run conda install --help 
#  and read through the output.

# How is the positional argument package_spec defined in
#  the documentation for conda install?
# R/ Packages to install into the conda environment.


# ### --------------------------------------------------------
# # # ------>>>>Install a conda package (II) 
# Installing a package is 
# largely a matter of listing 
# the name(s) of packages to 
# install after the command 
# conda install. But there is 
# more to it behind the scenes. 
# The versions of packages to 
# install (along with all their 
# dependencies) must be 
# compatible with all versions 
# of other software currently 
# installed. Often this 
# "satisfiability" constraint 
# depends on choosing a package 
# version compatible with a 
# particular version of Python 
# that is installed. Conda is 
# special among "package 
# managers" in that it always 
# guarantees this consistency; 
# you will see the phrase 
# "Solving environment..." 
# during installation to 
# indicate this computation. 

# For example, you may simply 
# instruct conda to install foo-
# lib. The tool first 
# determines which operating 
# system you are running, and 
# then narrows the match to 
# candidates made for this 
# platform. Then, conda 
# determines the version of 
# Python on the system (say 
# 3.7), and chooses the package 
# version for -py37. But, 
# beyond those simple limits, 
# all dependencies are checked. 

# Suppose foo-lib is available 
# in versions 1.0, 1.1, 1.2, 
# 2.0, 2.1, 2.2, 2.3, 3.0, 3.1 (
# for your platform and Python 
# version). As a first goal, 
# conda attempts to choose the 
# latest version of foo-lib. 
# However, maybe foo-lib 
# depends on bar-lib, which 
# itself is available in 
# various versions (say 1 
# through 20 in its versioning 
# scheme). It might be that foo-
# lib 3.1 is compatible with 
# bar-lib versions 17, 18, and 
# 19; but blob-lib (which is 
# already installed) is 
# compatible only with versions 
# of bar-lib less than 17. 
# Therefore, conda would 
# examine the compatibility of 
# foo-lib 3.0 as a fallback. In 
# this hypothetical, foo-lib 
# 3.0 is compatible with bar-
# lib 16, so that version is 
# chosen (bar-lib is also 
# updated to the latest 
# compatible version 16 in the 
# same command if an earlier 
# version is currently 
# installed).
conda install cytoolz
y

# ### --------------------------------------------------------
# # # ------>>>>What is semantic versioning? 
# Most Conda packages use a 
# system called semantic 
# versioning to identify 
# distinct versions of a 
# software package 
# unambiguously. Version labels 
# are usually chosen by the 
# project authors, not 
# necessarily the same people 
# who bundle the project as a 
# Conda package. There is no 
# technical requirement that a 
# project author's version 
# label coincides with a Conda 
# package version label, but 
# the convention of doing so is 
# almost always followed. 
# Semantic version labels can 
# be compared lexicographically 
# and make it easy to determine 
# which of two versions is the 
# later version. 

# Under semantic versioning, 
# software is labeled with a 
# three-part version identifier 
# of the form 
# MAJOR.MINOR.PATCH; the label 
# components are non-negative 
# integers separated by 
# periods. Assuming all 
# software starts at version 
# 0.0.0, the MAJOR version 
# number is increased when 
# significant new functionality 
# is introduced (often with 
# corresponding API changes). 
# Increases in the MINOR 
# version number generally 
# reflect improvements (e.g., 
# new features) that avoid 
# backward-incompatible API 
# changes. For instance, adding 
# an optional argument to a 
# function API (in a way that 
# allows old code to run 
# unchanged) is a change worthy 
# of increasing the MINOR 
# version number. An increment 
# to the PATCH version number 
# is appropriate mostly for bug 
# fixes that preserve the same 
# MAJOR and MINOR revision 
# numbers. Software patches do 
# not typically introduce new 
# features or change APIs at 
# all (except sometimes to 
# address security issues). 

# Many command-line tools 
# display their version 
# identifier by running tool --
# version. This information can 
# sometimes be displayed or 
# documented in other ways. For 
# example, suppose on some 
# system, a certain version of 
# Python is installed, and you 
# inquire about it like this: 

# python -c "import sys; 
# sys.version" '1.0.1 (Mar 26 
# 2014)' Looking at the output 
# above, which statement below 
# accurately characterizes the 
# semantic versioning of this 
# installed Python?
# The MAJOR version is 1, the PATCH is 1

# ### --------------------------------------------------------
# # # ------>>>>Which package version is installed?
conda list # then search for request version
2.22.0


# ### --------------------------------------------------------
# # # ------>>>>Install a specific version of a package (I)
conda install attrs=17.3
y


# ### --------------------------------------------------------
# # # ------>>>>Install a specific version of a package (II)
# Most commonly, 
# you'll use prefix-notation to 
# specify the package version(
# s) to install. But conda 
# offers even more powerful 
# comparison operations to 
# narrow versions. For example, 
# if you wish to install either 
# bar-lib versions 1.0, 1.4 or 
# 1.4.1b2, but definitely not 
# version 1.1, 1.2 or 1.3, you 
# could use: 

# conda install 'bar-lib=1.0|
# 1.4*' This may seem odd, but 
# you might know, for example, 
# that a bug was introduced in 
# 1.1 that wasn't fixed until 
# 1.4. You would prefer the 1.4 
# series, but, if it is 
# incompatible with other 
# packages, you can settle for 
# 1.0. Notice we have used 
# single quotes around the 
# version expression in this 
# case because several of the 
# symbols in these more complex 
# patterns have special 
# meanings in terminal shells. 
# It is easiest just to quote 
# them. 

# With conda you can also use 
# inequality comparisons to 
# select candidate versions (
# still resolving dependency 
# consistency). Maybe the bug 
# above was fixed in 1.3.5, and 
# you would like either the 
# latest version available (
# perhaps even 1.5 or 2.0 have 
# come out), but still avoiding 
# versions 1.1 through 1.3.4. 
# You could spell that as: 

# conda install 'bar-lib>1.3.4,<
# 1.1' For this exercise, 
# install the latest compatible 
# version of attrs that is 
# later than version 16, but 
# earlier than version 17.3. 
# Which version gets installed?
# R/ ---->17.2.0



# ### --------------------------------------------------------
# # # ------>>>>Update a conda package
conda update pandas
y


# ### --------------------------------------------------------
# # # ------>>>> Remove a conda package
conda remove pandas
y


# ### --------------------------------------------------------
# # # ------>>>> Search for available package versions?
conda search attrs


# ### --------------------------------------------------------
# # # ------>>>>Find dependencies for a 
# package version? The conda 
# search package_name --info 
# command reports a variety of 
# details about a specific 
# package. The syntax for 
# specifying just one version 
# is a little bit complex, but 
# prefix notation is allowed 
# here (just as you would with 
# conda install). 

# For example, running conda 
# search cytoolz=0.8.2 --info 
# will report on all available 
# package versions. As this 
# package has been built for a 
# variety of Python versions, a 
# number of packages will be 
# reported on. You can narrow 
# your query further with, 
# e.g.: 

# $ conda search 
# cytoolz=0.8.2=py36_0 --info 

# cytoolz 0.8.2 py36_0 <hr 
# />----------------- file 
# name   : cytoolz-0.8.2-
# py36_0.tar.bz2 name        : 
# cytoolz version     : 0.8.2 
# build string: py36_0 build 
# number: 0 channel     : https:
# //repo.anaconda.com/pkgs/free/osx-
# 64 size        : 352 KB 
# arch        : x86_64 
# constrains  : () 
# date        : 2016-12-23 
# license     : BSD 
# md5         : 
# cd6068b2389b1596147cc7218f0438fd 
# platform    : darwin 
# subdir      : osx-64 
# url         : https:
# //repo.anaconda.com/pkgs/free/osx-
# 64/cytoolz-0.8.2-
# py36_0.tar.bz2 dependencies: 
# python 3.6* toolz >=0.8.0 You 
# may use the * wildcard within 
# the match pattern. This is 
# often useful to match 
# 'foo=1.2.3=py36*' because 
# recent builds have attached 
# the hash of the build at the 
# end of the Python version 
# string, making the exact 
# match unpredictable. 

# Determine the dependencies of 
# the package numpy 1.13.1 with 
# Python 3.6.0 on your current 
# platform.
conda search numpy=1.13.1=py36* --info
# R/libgcc-ng >=7.2.0, libgfortran-ng >=7.2.0,<8.0a0, python >=3.6,<3.7.0a0, mkl >=2018.0.0,<2019.0a0, and blas * mkl


# ### --------------------------------------------------------
# # # ------>>>>Channels and why are they 
# needed? All Conda packages 
# we've seen so far were 
# published on the main or 
# default channel of Anaconda 
# Cloud. A Conda channel is an 
# identifier of a path (e.g., 
# as in a web address) from 
# which Conda packages can be 
# obtained. Using the public 
# cloud, installing without 
# specifying a channel points 
# to the main channel at https:
# //repo.anaconda.com/pkgs/main; 
# where hundreds of packages 
# are available. Although 
# covering a wide swath, the 
# main channel contains only 
# packages that are (
# moderately) curated by 
# Anaconda Inc. Given finite 
# resources and a particular 
# area focus, not all genuinely 
# worthwhile packages are 
# vetted by Anaconda Inc. 

# If you happen to be working 
# in a firewalled or airgapped 
# environment with a private 
# installation of Anaconda 
# Repository, your default 
# channel may point to a 
# different (internal) URL, but 
# the same concepts will apply. 

# Anyone may register for an 
# account with Anaconda Cloud, 
# thereby creating their own 
# personal Conda channel. This 
# is covered in the companion 
# course Conda for Building & 
# Distributing Packages (along 
# with creating and uploading 
# your own packages). For this 
# course, just understand that 
# many users have accounts and 
# corresponding channels. 

# Which description best 
# characterizes Conda channels?
# R/Channels are a means for a user to publish packages independently.



# ### --------------------------------------------------------
# # # ------>>>>Searching within channels If 
# a particular colleague or 
# other recognized user may 
# have published a package 
# useful to you, you can search 
# for it using the anaconda 
# search command. For example, 
# David Mertz, the principal 
# author of this course, has a 
# channel and Anaconda Cloud 
# account called davidmertz. 
# You can search his channel 
# using the command below; the 
# option --channel (or -c for 
# short) specifies the channel 
# to search. Particular users 
# may have published more niche 
# software you would like to 
# use; for example, colleagues 
# of yours may publish packages 
# of special use in your field 
# or topic area. 

# $ conda search --channel 
# davidmertz --override-
# channels --platform linux-64 
# Loading channels: done 
# Name                       
# Version                   
# Build  Channel 
# accelerate                 
# 2.2.0               
# np110py27_2  davidmertz 
# accelerate                 
# 2.2.0               
# np110py35_2  davidmertz 
# accelerate-dldist          
# 0.1                 
# np110py27_1  davidmertz 
# accelerate-dldist          
# 0.1                 
# np110py35_1  davidmertz 
# accelerate-gensim          
# 0.12.3             
# np110py27_96  davidmertz 
# accelerate-gensim          
# 0.12.3             
# np110py35_96  davidmertz 
# accelerate-skimage         
# 0.1.0                    
# py27_1  davidmertz accelerate-
# skimage         
# 0.1.0                    
# py35_1  davidmertz 
# constants                  
# 0.0.2                    
# py35_0  davidmertz 
# humidity                   
# 0.1              
# py36ha038022_0  davidmertz 
# textadapter                
# 2.0.0                    
# py27_0  davidmertz 
# textadapter                
# 2.0.0                    
# py35_0  davidmertz 
# textadapter                
# 2.0.0                    
# py36_0  davidmertz In this 
# case, the switch --override-
# channels is used to prevent 
# searching on default 
# channels. The switch --
# platform is used to select a 
# platform that may differ from 
# the one on which the search 
# is run (absent the switch, 
# the current computer's 
# platform is used). 

# The first search is unusual 
# in that it does not specify a 
# package name, which is more 
# typical actual use. For 
# example, you might want to 
# know which versions of the 
# package of textadapter for 
# the win-64 platform are 
# available for any version of 
# Python (assuming you know in 
# which channels to look): 

# $ conda search -c conda-
# forge -c sseefeld -c 
# gbrener --platform win-64 
# textadapter Loading channels: 
# done 
# Name                       
# Version                   
# Build  Channel 
# textadapter                
# 2.0.0                    
# py27_0  conda-forge 
# textadapter                
# 2.0.0                    
# py27_0  sseefeld 
# textadapter                
# 2.0.0                    
# py34_0  sseefeld 
# textadapter                
# 2.0.0                    
# py35_0  conda-forge 
# textadapter                
# 2.0.0                    
# py35_0  sseefeld 
# textadapter                
# 2.0.0                    
# py36_0  sseefeld 



# Based on the examples shown, 
# in which of the channels used 
# in the examples above could 
# you find an osx-64 version of 
# textadapter for Python 3.6?
conda search -c conda-forge -c sseefeld -c gbrener -c davidmertz --platform osx-64 textadapter
# R/davidmertz


# ### --------------------------------------------------------
# # # ------>>>>Searching across channels
anaconda search boltons
# 17.1.0 for me it's the right one 
# but answer is 20.0.0

# ### --------------------------------------------------------
# # # ------>>>> Default, non-default, and special channels
conda search -c conda-forge | grep conda-forge | wc -l
211557
# R/ About 100 thousand



# ### --------------------------------------------------------
# # # ------>>>> Installing from a channel
conda install --channel conda-forge youtube-dl
y
conda list


# ### --------------------------------------------------------
# # # ------>>>>Environments and why are they 
# needed? Conda environments 
# allow multiple incompatible 
# versions of the same (
# software) package to coexist 
# on your system. An 
# environment is simply a file 
# path containing a collection 
# of mutually compatible 
# packages. By isolating 
# distinct versions of a given 
# package (and their 
# dependencies) in distinct 
# environments, those versions 
# are all available to work on 
# particular projects or tasks. 

# There are a large number of 
# reasons why it is best 
# practice to use environments, 
# whether as a data scientist, 
# software developer, or domain 
# specialist. Without the 
# concept of environments, 
# users essentially rely on and 
# are restricted to whichever 
# particular package versions 
# are installed globally (or in 
# their own user accounts) on a 
# particular machine. Even when 
# one user moves scripts 
# between machines (or shares 
# them with a colleague), the 
# configuration is often 
# inconsistent in ways that 
# interfere with seamless 
# functionality. Conda 
# environments solve both these 
# problems. You can easily 
# maintain and switch between 
# as many environments as you 
# like, and each one has 
# exactly the collection of 
# packages that you want. 

# For example, you may develop 
# a project comprising scripts, 
# notebooks, libraries, or 
# other resources that depend 
# on a particular collection of 
# package versions. You later 
# want to be able to switch 
# flexibly to newer versions of 
# those packages and to ensure 
# the project continues to 
# function properly before 
# switching wholly. Or 
# likewise, you may want to 
# share code with colleagues 
# who are required to use 
# certain package versions. In 
# this context, an environment 
# is a way of documenting a 
# known set of packages that 
# correctly support your 
# project. 

# Which statement is true of 
# Conda environments?
# R/Conda environments allow for flexible version management of packages.




# ### --------------------------------------------------------
# # # ------>>>>Which environment am I using?
# When using conda, you are 
# always in some environment, 
# but it may be the default (
# called the base or root 
# environment). Your current 
# environment has a name and 
# contains a collection of 
# packages currently associated 
# with that environment. There 
# are a few ways to determine 
# the current environment. 

# Most obviously, at a terminal 
# prompt, the name of the 
# current environment is 
# usually prepended to the rest 
# of your prompt in 
# parentheses. Alternatively, 
# the subcommand conda env list 
# displays a list of all 
# environments on your current 
# system; the currently 
# activated one is marked with 
# an asterisk in the middle 
# column. The subcommands of 
# conda env (sometimes with 
# suitable switches) encompass 
# most of your needs for 
# working with environments. 

# The output of conda env list 
# shows that each environment 
# is associated with a 
# particular directory. This is 
# not the same as your current 
# working directory for a given 
# project; being "in" an 
# environment is completely 
# independent of the directory 
# you are working in. Indeed, 
# you often wish to preserve a 
# certain Conda environment and 
# edit resources across 
# multiple project directories (
# all of which rely on the same 
# environment). The environment 
# directory displayed by conda 
# env list is simply the top-
# level file path in which all 
# resources associated with 
# that environment are stored; 
# you need never manipulate 
# those environment directories 
# directly (other than via the 
# conda command); indeed, it is 
# much safer to leave those 
# directories alone! 

# For example, here is output 
# you might see in a particular 
# terminal: 

# (test-project) $ conda env 
# list # conda environments: # 
# base                     
# /home/repl/miniconda 
# py1.0                    
# /home/repl/miniconda/envs/py1.0 
# stats-research           
# /home/repl/miniconda/envs/stats-
# research test-
# project          *  
# /home/repl/miniconda/envs/test-
# project Following the example 
# above, what is the name of 
# the environment you are using 
# in the current session? Even 
# if you determine the answer 
# without running a command, 
# run conda env list to get a 
# feel of using that subcommand.
conda env list
#R/ course-project



# ### --------------------------------------------------------
# # # # ------>>>>What packages are installed 
# in an environment? (I) The 
# command conda list seen 
# previously displays all 
# packages installed in the 
# current environment. You can 
# reduce this list by appending 
# the particular package you 
# want as an option. The 
# package can be specified 
# either as a simple name, or 
# as a regular expression 
# pattern. This still displays 
# the version (and channel) 
# associated with the installed 
# package(s). For example: 

# (test-env) $ conda list 
# 'numpy|pandas' # packages in 
# environment at 
# /home/repl/miniconda/envs/test-
# env: # # 
# Name                    
# Version                   
# Build  Channel 
# numpy                     
# 1.11.3                   
# py35_0 
# pandas                    
# 0.18.1              
# np111py35_0 Without 
# specifying 'numpy|pandas', 
# these same two lines would be 
# printed, simply interspersed 
# with many others for the 
# various other installed 
# packages. Notice that the 
# output displays the filepath 
# associated with the current 
# environment first: in this 
# case, 
# /home/repl/miniconda/envs/test-
# env as test-env is the active 
# environment (as also shown at 
# the prompt). 

# Following this example, what 
# versions of numpy and pandas 
# are installed in the current (
# base/root) environment?
conda list 'numpy|pandas'
# R/ numpy=1.16.0; pandas=0.22.0


# ### --------------------------------------------------------
# # # ------>>>>What packages are installed 
# in an environment? (II) It is 
# often useful to query a 
# different environment's 
# configuration (i.e., as 
# opposed to the currently 
# active environment). You 
# might do this simply to 
# verify the package versions 
# in that environment that you 
# need for a given project. Or 
# you may wish to find out what 
# versions you or a colleague 
# used in some prior project (
# developed in that other 
# environment). The switch --
# name or -n allows you to 
# query another environment. 
# For example, 

# (course-env) $ conda list --
# name test-env 'numpy|pandas' 
# # packages in environment at 
# /home/repl/miniconda/envs/test-
# env: # # 
# Name                    
# Version                   
# Build  Channel 
# numpy                     
# 1.11.3                   
# py35_0 
# pandas                    
# 0.18.1              
# np111py35_0 Without 
# specifying the --name 
# argument, the command conda 
# list would run in the current 
# environment. The output would 
# then be the versions of numpy 
# and pandas present in the 
# current environment. 

# Suppose you created an 
# environment called pd-2015 in 
# 2015 when you were working on 
# a project. Identify which 
# versions of numpy and pandas 
# were installed in the 
# environment pd-2015.
conda list -n pd-2015 'numpy|pandas'
# R/ numpy=1.16.4; pandas=0.22.0


# ### --------------------------------------------------------
# # # ------>>>> Switch between environments
conda activate course-env
conda activate pd-2015
conda deactivate


# ### --------------------------------------------------------
# # # ------>>>> Remove an environment
conda env remove -n deprecated

# ### --------------------------------------------------------
# # # ------>>>>Create a new environment
conda create -n conda-essentials attrs=19.1.0 cytoolz
y
conda activate conda-essentials
conda list


# ### --------------------------------------------------------
# # # ------>>>>Export an environment
conda env export -n course-env -f course-env.yml


# ### --------------------------------------------------------
# # # ------>>>>Create an environment from a shared specification
conda env create --file environment.yml
conda env create -f shared-config.yml


# ### --------------------------------------------------------
# # # ------>>>>Compatibility with different versions
cat weekly_humidity.py
python weekly_humidity.py
conda activate pd-2015
python weekly_humidity.py

# ### --------------------------------------------------------
# # # ------>>>>Updating a script
nano weekly_humidity.py
print(humidity.rolling(7).mean().tail(5))
# control + x
#  enter
python weekly_humidity.py
conda activate pd-2015
python weekly_humidity.py
