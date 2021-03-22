# Environment Installation 
## ----->                 Power Shell 

# to print all the packages 
# of the environment and saved it 
pip freeze > filename.txt

# to install all the dependencies in a file 
pip install -r filename.txt

# to install ipykernel 
python -m ipykernel install --user --name environment_name --display-namedisplay-name "what-do-you-want-to-display"
 
# to update env with a yml file 
conda env update -f file_with_versions.yml --debug 

# to create a env with a single command 
conda env create --name environment_name --file="path/file_with_versions.yml" --name environment_name -v --force

# show version of a package 
pip show package_name

# clone env with a file 
conda create --name environment_name --clone azureml_py36 --debug


# Fix some problems with scikit-learn 
pip show scikit-learn 
pip unistall scikit-learn -y
pip show scikit-learn 
pip unistall scikit-learn -y

# Gives what env has
conda deactivate && conda env list

# Create a environment
conda create --name environment_name 
conda activate environment_name
conda env update --name environment_name --file="path/file_with_versions.yml" --prune --debug -v
# and shut down the vm  
# open again session 
# check scikit learn and azureml-sdk

