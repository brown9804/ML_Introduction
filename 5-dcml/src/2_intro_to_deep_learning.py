
# #--  --  --  -- Introduction to Deep Learning in Python
# # Used for Data Scientist Training Path 
# #FYI it's a compilation of how to work
# #with different commands.


# ### --------------------------------------------------------
# # # # ------>>>> Comparing neural network models to classical regression models
# Which of the models in the diagrams has greater ability to account for interactions?
# R/ Model 2
# Because ---> Incorrect. Each node adds to the model's ability to capture interactions. So the more nodes you have, the more interactions you can capture.



# ### --------------------------------------------------------
# # # # ------>>>> Coding the forward propagation algorithm
# Calculate node 0 value: node_0_value
node_0_value = (input_data * weights['node_0']).sum()

# Calculate node 1 value: node_1_value
node_1_value = (input_data * weights['node_1']).sum()

# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = np.array([node_0_value, node_1_value])

# Calculate output: output
output = (hidden_layer_outputs * weights['output']).sum()

# Print output
print(output)




# ### --------------------------------------------------------
# # # # ------>>>> The Rectified Linear Activation Function
def relu(input):
    '''Define your relu activation function here'''
    # Calculate the value for the output of the relu function: output
    output = max(0, input)
    
    # Return the value just calculated
    return(output)

# Calculate node 0 value: node_0_output
node_0_input = (input_data * weights['node_0']).sum()
node_0_output = relu(node_0_input)

# Calculate node 1 value: node_1_output
node_1_input = (input_data * weights['node_1']).sum()
node_1_output = relu(node_1_input)

# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = np.array([node_0_output, node_1_output])

# Calculate model output (do not apply relu)
model_output = (hidden_layer_outputs * weights['output']).sum()

# Print model output
print(model_output)




# ### --------------------------------------------------------
# # # # ------>>>> Applying the network to many observations/rows of data
# Define predict_with_network()
def predict_with_network(input_data_row, weights):

    # Calculate node 0 value
    node_0_input = (input_data_row * weights['node_0']).sum()
    node_0_output = relu(node_0_input)

    # Calculate node 1 value
    node_1_input = (input_data_row * weights['node_1']).sum()
    node_1_output = relu(node_1_input)

    # Put node values into array: hidden_layer_outputs
    hidden_layer_outputs = np.array([node_0_output, node_1_output])
    
    # Calculate model output
    input_to_final_layer = (hidden_layer_outputs * weights['output']).sum()
    model_output = relu(input_to_final_layer)
    
    # Return model output
    return(model_output)


# Create empty list to store prediction results
results = []
for input_data_row in input_data:
    # Append prediction to results
    results.append(predict_with_network(input_data_row, weights))

# Print results
print(results)




# ### --------------------------------------------------------
# # # # ------>>>> Forward propagation in a deeper network
# You now have a model with 2 hidden layers. The values for 
# an input data point are shown inside the input nodes. 
# The weights are shown on the edges/lines. What prediction 
# would this model make on this data point?

# Assume the activation function at each node
#  is the identity function. That is, each node's 
#  output will be the same as its input. So the value of 
#  the bottom node in the first hidden layer is -1, and not 0, 
#  as it would be if the ReLU activation function was used.
# R/ 0



# ### --------------------------------------------------------
# # # # ------>>>> Multi-layer neural networks
def predict_with_network(input_data):
    # Calculate node 0 in the first hidden layer
    node_0_0_input = (input_data * weights['node_0_0']).sum()
    node_0_0_output = relu(node_0_0_input)

    # Calculate node 1 in the first hidden layer
    node_0_1_input = (input_data * weights['node_0_1']).sum()
    node_0_1_output = relu(node_0_1_input)

    # Put node values into array: hidden_0_outputs
    hidden_0_outputs = np.array([node_0_0_output, node_0_1_output])

    # Calculate node 0 in the second hidden layer
    node_1_0_input = (hidden_0_outputs * weights['node_1_0']).sum()
    node_1_0_output = relu(node_1_0_input)

    # Calculate node 1 in the second hidden layer
    node_1_1_input = (hidden_0_outputs * weights['node_1_1']).sum()
    node_1_1_output = relu(node_1_1_input)

    # Put node values into array: hidden_1_outputs
    hidden_1_outputs = np.array([node_1_0_output, node_1_1_output])
    
    # Calculate output here: model_output
    model_output = (hidden_1_outputs * weights['output']).sum()
    
    # Return model_output
    return(model_output)

output = predict_with_network(input_data)
print(output)




# ### --------------------------------------------------------
# # # # ------>>>> Representations are learned
# How are the weights that determine the features/interactions in Neural Networks created?
# R/ The model training process sets them to optimize predictive accuracy.




# ### --------------------------------------------------------
# # # # ------>>>> Levels of representation
# Which layers of a model capture more complex or "higher level" interactions?
# R/ The last layers capture the most complex interactions.




# ### --------------------------------------------------------
# # # # ------>>>> Calculating model errors
# For the exercises in this 
# chapter, you'll continue 
# working with the network to 
# predict transactions for a 
# bank. What is the error (
# predicted - actual) for the 
# following network using the 
# ReLU activation function when 
# the input data is [3, 2] and 
# the actual value of the 
# target (what you are trying 
# to predict) is 5? It may be 
# helpful to get out a pen and 
# piece of paper to calculate 
# these values.
# R/ 11 
# Because ---> Well done! The network generates a prediction of 16, which results in an error of 11.



# ### --------------------------------------------------------
# # # # ------>>>> Understanding how weights change model accuracy
# R/ Less accurate
# Because ---> Increasing the weight to 2.01 would increase the resulting error from 9 to 9.08, making the predictions less accurate.




# ### --------------------------------------------------------
# # # # ------>>>> Coding how weight changes affect accuracy
# The data point you will make a prediction for
input_data = np.array([0, 3])

# Sample weights
weights_0 = {'node_0': [2, 1],
             'node_1': [1, 2],
             'output': [1, 1]
            }

# The actual target value, used to calculate the error
target_actual = 3

# Make prediction using original weights
model_output_0 = predict_with_network(input_data, weights_0)

# Calculate error: error_0
error_0 = model_output_0 - target_actual

# Create weights that cause the network to make perfect prediction (3): weights_1
weights_1 = {'node_0': [2, 1],
             'node_1': [1, 2],
             'output': [1, 0]
            }

# Make prediction using new weights: model_output_1
model_output_1 = predict_with_network(input_data, weights_1)

# Calculate error: error_1
error_1 = model_output_1 - target_actual

# Print error_0 and error_1
print(error_0)
print(error_1)




# ### --------------------------------------------------------
# # # # ------>>>> Scaling up to multiple data points
from sklearn.metrics import mean_squared_error

# Create model_output_0 
model_output_0 = []
# Create model_output_0
model_output_1 = []

# Loop over input_data
for row in input_data:
    # Append prediction to model_output_0
    model_output_0.append(predict_with_network(row, weights_0))
    
    # Append prediction to model_output_1
    model_output_1.append(predict_with_network(row, weights_1))

# Calculate the mean squared error for model_output_0: mse_0
mse_0 = mean_squared_error(target_actuals, model_output_0)

# Calculate the mean squared error for model_output_1: mse_1
mse_1 = mean_squared_error(target_actuals, model_output_1)

# Print mse_0 and mse_1
print("Mean squared error with weights_0: %f" %mse_0)
print("Mean squared error with weights_1: %f" %mse_1)




# ### --------------------------------------------------------
# # # # ------>>>> Calculating slopes
# Calculate the predictions: preds
preds = (weights * input_data).sum()

# Calculate the error: error
error = preds - target

# Calculate the slope: slope
slope = input_data * error * 2

# Print the slope
print(slope)




# ### --------------------------------------------------------
# # # # ------>>>> Improving model weights
# Set the learning rate: learning_rate
learning_rate = 0.01

# Calculate the predictions: preds
preds = (weights * input_data).sum()

# Calculate the error: error
error = preds - target

# Calculate the slope: slope
slope = 2 * input_data * error

# Update the weights: weights_updated
weights_updated = weights - learning_rate * slope

# Get updated predictions: preds_updated
preds_updated = (weights_updated * input_data).sum()

# Calculate updated error: error_updated
error_updated = preds_updated - target

# Print the original error
print(error)

# Print the updated error
print(error_updated)




# ### --------------------------------------------------------
# # # # ------>>>> Making multiple updates to weights
n_updates = 20
mse_hist = []

# Iterate over the number of updates
for i in range(n_updates):
    # Calculate the slope: slope
    slope = get_slope(input_data, target, weights)
    
    # Update the weights: weights
    weights = weights - 0.01 * slope
    
    # Calculate mse with new weights: mse
    mse = get_mse(input_data, target, weights)
    
    # Append the mse to mse_hist
    mse_hist.append(mse)

# Plot the mse history
plt.plot(mse_hist)
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.show()




# ### --------------------------------------------------------
# # # # ------>>>> The relationship between forward and backward propagation
# If you have gone through 4 
# iterations of calculating 
# slopes (using backward 
# propagation) and then updated 
# weights, how many times must 
# you have done forward 
# propagation?

# R/ 4 
# Because ---> Each time you generate predictions using forward propagation, you update the weights using backward propagation.




# ### --------------------------------------------------------
# # # # ------>>>> Thinking about backward propagation
# If your predictions were all 
# exactly right, and your 
# errors were all exactly 0, 
# the slope of the loss 
# function with respect to your 
# predictions would also be 0. 
# In that circumstance, which 
# of the following statements 
# would be correct?
# R/ The updates to all weights in the network would also be 0.




# ### --------------------------------------------------------
# # # # ------>>>> A round of backpropagation
# In the network shown below, 
# we have done forward 
# propagation, and node values 
# calculated as part of forward 
# propagation are shown in 
# white. The weights are shown 
# in black. Layers after the 
# question mark show the slopes 
# calculated as part of back-
# prop, rather than the forward-
# prop values. Those slope 
# values are shown in purple. 
# This network again uses the 
# ReLU activation function, so 
# the slope of the activation 
# function is 1 for any node 
# receiving a positive value as 
# input. Assume the node being 
# examined had a positive 
# value (so the activation 
# function's slope is 1).
# R/ 6 



# ### --------------------------------------------------------
# # # # ------>>>> Understanding your data
# You will soon start building models in 
# Keras to predict wages based on various 
# professional and demographic factors. 
# Before you start building a model, it's 
# good to understand your data by 
# performing some exploratory analysis. 
# The data is pre-loaded into a pandas 
# DataFrame called df. Use the .head() 
# and .describe() methods in the IPython 
# Shell for a quick overview of the 
# DataFrame. The target variable you'll 
# be predicting is wage_per_hour. Some of 
# the predictor variables are binary 
# indicators, where a value of 1 
# represents True, and 0 represents 
# False. Of the 9 predictor variables in 
# the DataFrame, how many are binary 
# indicators? The min and max values as 
# shown by .describe() will be 
# informative here. How many binary 
# indicator predictors are there?
df.decribe()
df.head()
# R/ 6 


# ### --------------------------------------------------------
# # # # ------>>>> Specifying a model
# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

# Set up the model: model
model = Sequential()

# Add the first layer
model.add(Dense(50, activation='relu', input_shape=(n_cols,)))

# Add the second layer
model.add(Dense(32, activation='relu'))

# Add the output layer
model.add(Dense(1))




# ### --------------------------------------------------------
# # # # ------>>>> Compiling the model
# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential

# Specify the model
n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Verify that model contains information from compiling
print("Loss function: " + model.loss)




# ### --------------------------------------------------------
# # # # ------>>>> Fitting the model
# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential

# Specify the model
n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model
model.fit(predictors, target)




# ### --------------------------------------------------------
# # # # ------>>>> Understanding your classification data
# Now you will start modeling 
# with a new dataset for a 
# classification problem. This 
# data includes information 
# about passengers on the 
# Titanic. You will use 
# predictors such as age, fare 
# and where each passenger 
# embarked from to predict who 
# will survive. This data is 
# from a tutorial on data 
# science competitions. Look 
# here for descriptions of the 
# features. The data is pre-
# loaded in a pandas DataFrame 
# called df. It's smart to 
# review the maximum and 
# minimum values of each 
# variable to ensure the data 
# isn't misformatted or 
# corrupted. What was the 
# maximum age of passengers on 
# the Titanic? Use the 
# .describe() method in the 
# IPython Shell to answer this 
# question.
df.describe()
# R/ 80 

# ### --------------------------------------------------------
# # # # ------>>>> Last steps in classification models
# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

# Convert the target to categorical: target
target = to_categorical(df.survived)

# Set up the model
model = Sequential()

# Add the first layer
model.add(Dense(32, activation='relu', input_shape=(n_cols,)))

# Add the output layer
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(predictors, target)




# ### --------------------------------------------------------
# # # # ------>>>> Making predictions
# Specify, compile, and fit the model
model = Sequential()
model.add(Dense(32, activation='relu', input_shape = (n_cols,)))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='sgd', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(predictors, target)

# Calculate predictions: predictions
predictions = model.predict(pred_data)

# Calculate predicted probability of survival: predicted_prob_true
predicted_prob_true = predictions[:,1]

# print predicted_prob_true
print(predicted_prob_true)




# ### --------------------------------------------------------
# # # # ------>>>> Diagnosing optimization problems
# Which of the following could prevent a model from showing an improved loss in its first few epochs?
# R/ All of the above 
# Learning rate too low
# Learning rate too high.
# Poor choice of activation function.


# ### --------------------------------------------------------
# # # # ------>>>> Changing optimization parameters
# Import the SGD optimizer
from keras.optimizers import SGD

# Create list of learning rates: lr_to_test
lr_to_test = [.000001, 0.01, 1]

# Loop over learning rates
for lr in lr_to_test:
    print('\n\nTesting model with learning rate: %f\n'%lr )
    
    # Build new model to test, unaffected by previous models
    model = get_new_model()
    
    # Create SGD optimizer with specified learning rate: my_optimizer
    my_optimizer = SGD(lr=lr)
    
    # Compile the model
    model.compile(optimizer = my_optimizer, loss = 'categorical_crossentropy')
    
    # Fit the model
    model.fit(predictors, target)


# ### --------------------------------------------------------
# # # # ------>>>> Evaluating model accuracy on validation dataset
# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]
input_shape = (n_cols,)

# Specify the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

# Fit the model
hist = model.fit(predictors, target, validation_split=0.3)


# ### --------------------------------------------------------
# # # # ------>>>> Early stopping: Optimizing the optimization
# Import EarlyStopping
from keras.callbacks import EarlyStopping

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]
input_shape = (n_cols,)

# Specify the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Fit the model
model.fit(predictors, target, validation_split=0.3, epochs=30, callbacks=[early_stopping_monitor])


# ### --------------------------------------------------------
# # # # ------>>>> Experimenting with wider networks
# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Create the new model: model_2
model_2 = Sequential()

# Add the first and second layers
model_2.add(Dense(100, activation='relu', input_shape=input_shape))
model_2.add(Dense(100, activation='relu'))

# Add the output layer
model_2.add(Dense(2, activation='softmax'))

# Compile model_2
model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit model_1
model_1_training = model_1.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)

# Fit model_2
model_2_training = model_2.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)

# Create the plot
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()


# ### --------------------------------------------------------
# # # # ------>>>> Adding layers to a network
# The input shape to use in the first hidden layer
input_shape = (n_cols,)

# Create the new model: model_2
model_2 = Sequential()

# Add the first, second, and third hidden layers
model_2.add(Dense(50, activation='relu', input_shape=input_shape))
model_2.add(Dense(50, activation='relu'))
model_2.add(Dense(50, activation='relu'))

# Add the output layer
model_2.add(Dense(2, activation='softmax'))

# Compile model_2
model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit model 1
model_1_training = model_1.fit(predictors, target, epochs=20, validation_split=0.4, callbacks=[early_stopping_monitor], verbose=False)

# Fit model 2
model_2_training = model_2.fit(predictors, target, epochs=20, validation_split=0.4, callbacks=[early_stopping_monitor], verbose=False)

# Create the plot
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()





# ### --------------------------------------------------------
# # # # ------>>>> Experimenting with model structures
# You've just run an experiment where you 
# compared two networks that were 
# identical except that the 2nd network 
# had an extra hidden layer. You see that 
# this 2nd network (the deeper network) 
# had better performance. Given that, 
# which of the following would be a good 
# experiment to run next for even better 
# performance?
# R/Use more units in each hidden layer.
#  Because ---> Increasing the number of units in each hidden layer would be a good next step to try achieving even better performance.



# ### --------------------------------------------------------
# # # # ------>>>> Building your own digit recognition model
# Create the model: model
model = Sequential()

# Add the first hidden layer
model.add(Dense(50, activation='relu', input_shape=(784,)))

# Add the second hidden layer
model.add(Dense(50, activation='relu', input_shape=(784,)))

# Add the output layer
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(X, y, validation_split=0.3)

