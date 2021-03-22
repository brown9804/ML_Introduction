# [Notes] Azure Machine Learning Environments 

## References
[1] From https://docs.microsoft.com/en-us/azure/machine-learning/classic/ <br/>
[2] From https://naadispeaks.wordpress.com/2020/11/30/different-computation-options-on-azure-machine-learning/#:~:text=Compute%20clusters%20are%20different%20from,using%20parallel%20processing%20for%20computations <br/>
[3] From https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?tabs=python <br/>
[4] From https://www.digitalcrafts.com/blog/student-blog-what-postman-and-why-use-it

## Experiments 
It's a training script with different config 

## Model 
It's a algorithm that run the traing script(experiment). A deployed experiment is a registered model, so it's like the output of the experiment. 

## Endpoint 
It's a deployed model (web service / API). It's important to review the security configuration, considering ports and if there is any VN. 

## Compute instance 
`-> Workspace ... (like vm)`

Azure Compute instances offer fully managed 
virtual machines loaded with most of the essential 
frameworks /libraries for performing machine learning 
and data science experiments. 

## Compute cluster 
`-> Where is scala running the ml training model` <br/>
Azure Machine Learning compute cluster is a managed-compute infrastructure that allows you to easily create a single or multi-node compute. The compute executes in a containerized environment and packages your model dependencies in a Docker container. Compute clusters can run jobs securely in a virtual network environment, without requiring enterprises to open up SSH ports. The job executes in a containerized environment and packages your model dependencies in a Docker container. <br/>
Compute clusters are different from compute instances with their ability of having one or more compute nodes. These compute nodes can be created with our desired hardware configurations. Why having more than one node? That comes with the ability of using parallel processing for computations. If you are going do to hyperparameter tuning/ GPU based complex computations/ several machine learning runs at once you may have to create a compute cluster. If you are running Automated Machine Learning expriment with AzureML, you must have a compute cluster to perform computations.


## Infence cluster 
`-> For endpoints `

There are two options to deploy Azure machine learning web services as REST endpoints. <br/>
1) Use ACI (Azure Container Instances) 
2) Use AKS (Azure Kubernetes Service)

## PostMan
To connect to a web service
