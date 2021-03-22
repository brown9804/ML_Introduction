# This is my compilations of 
# several tips/information 
# about Azure ML env 


# Experiments -> training script with different config 
# a deployed experiment is a registered model -> Output of the experiment <-> Algorithm that run the traing script(experiment)
# a deployed model is an endpoint 
# endpoint <-> web service / api

From https://naadispeaks.wordpress.com/2020/11/30/different-computation-options-on-azure-machine-learning/#:~:text=Compute%20clusters%20are%20different%20from,using%20parallel%20processing%20for%20computations.

# Compute instance -> workspace ... (like vm)
Azure Compute instances offer fully managed 
virtual machines loaded with most of the essential 
frameworks /libraries for performing machine learning 
and data science experiments. 

# Compute cluster -> Where is scala running the ml training model 
Compute clusters are different from compute instances with their ability of having one or more compute nodes. These compute nodes can be created with our desired hardware configurations.
Why having more than one node? That comes with the ability of using parallel processing for computations. If you are going do to hyperparameter tuning/ GPU based complex computations/ several machine learning runs at once you may have to create a compute cluster.
If you are running Automated Machine Learning expriment with AzureML, you must have a compute cluster to perform computations.


# Infence cluster -> for endpoints 
There are two options to deploy Azure machine learning web services as REST endpoints. 
1) Use ACI (Azure Container Instances)
2) Use AKS (Azure Kubernetes Service)

# Endpoints 
It's important to review the security configuration, considering ports and if there is any VN 

# PostMan
To connect to a web service