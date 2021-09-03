
# coding: utf-8

# # LASSO using Coordinate Descent

# In[171]:

import pandas as pd
import numpy as np


# In[169]:


dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 
              'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 
              'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 
              'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 
              'id':str, 'sqft_lot':int, 'view':int}

sales = pd.read_csv('kc_house_data.csv', dtype = dtype_dict)
train = pd.read_csv('kc_house_train_data.csv', dtype=dtype_dict)
test = pd.read_csv('kc_house_test_data.csv', dtype=dtype_dict)


# ## convert dataframe into matrix

# In[170]:

#convert into a matrix
def convert_into_numpy_matrix(data_frame, features, output):
    data_frame['constant'] = 1
    features = ['constant'] + features
    features_df = data_frame[features]
    feature_matrix = features_df.as_matrix()
    output_df = data_frame[output]
    return (feature_matrix, output_df)

#predicting outputs
def predict_output(features, weights):
    predictions = np.dot(features, weights)
    return predictions


# ## normalizing the features matrix

# In[131]:

#nomalizing features
def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix, axis=0)
    normalized_features = feature_matrix/norms
    return (normalized_features, norms)


# In[133]:

#testing the normalize_features() function
features, norms = normalize_features(np.array([[3.,6.,9.],[4.,8.,12.]]))
print features
# should print
# [[ 0.6  0.6  0.6]
#  [ 0.8  0.8  0.8]]
print norms
# should print
# [5.  10.  15.]


# In[82]:

#simple model with 2 features
simple_features = ['sqft_living', 'bedrooms']
(simple_features_matrix, op) = convert_into_numpy_matrix(sales, simple_features, 'price')
(simple_features_matrix, norms_simple_matrix) = normalize_features(simple_features_matrix) 

initial_weights = np.array([1., 4., 1.])
predictions1 = predict_output(simple_features_matrix, initial_weights)

We now need to implement the following part:

       ┌ (rho[i] + lambda/2)    if rho[i] < -lambda/2
w[i] = ├ 0                      if -lambda/2 <= rho[i] <= lambda/2
       └ (rho[i] - lambda/2)    if rho[i] > lambda/2

rho[i] = SUM[ [feature_i]*(output - prediction + w[i]*[feature_i]) ]
# ## function to calcuate value of rho

# In[83]:

#calculate value of rho
def calculate_rho(features_matrix, output, weights, i):
    predictions = predict_output(features_matrix, initial_weights)
    feature_i = features_matrix[:, i]
    temp = (output-predictions)+(weights[i]*feature_i) 
    rho_i = sum(feature_i*temp)
    return rho_i


# In[90]:

from decimal import Decimal


rho_1 = calculate_rho(simple_features_matrix, op, initial_weights, 1)
rho_2 = calculate_rho(simple_features_matrix, op, initial_weights, 2)
print 'rho1: ', '%.2E' % Decimal(rho_1), "& rho2: ",'%.2E' % Decimal(rho_2)
print 'rho1: ',rho_1, "& rho2: ",rho_2


# ## lasso coordinate descent

# In[8]:

#calculate value of rho with predictions
def calculate_rho2(features_matrix, output, weights, i, predictions):
    feature_i = features_matrix[:, i]
    temp = (output-predictions)+(weights[i]*feature_i) 
    rho_i = sum(feature_i*temp)
    return rho_i

#lasso coordinate descent algo
def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    predictions = predict_output(feature_matrix, weights)
    rho_i = calculate_rho2(feature_matrix, output, weights, i, predictions)
    
    if i == 0:
        new_weight_i = rho_i
    elif rho_i < (-l1_penalty/2):
        new_weight_i = rho_i + (l1_penalty/2)
    elif rho_i > l1_penalty/2:
        new_weight_i = rho_i - (l1_penalty/2)
    else:
        new_weight_i = 0
        
    return new_weight_i


# In[130]:

#testing the lasso coordinate descent algo
# should print 0.425558846691
import math
print lasso_coordinate_descent_step(1, np.array([[3./math.sqrt(13),1./math.sqrt(10)],
                   [2./math.sqrt(13),3./math.sqrt(10)]]), np.array([1., 1.]), np.array([1., 4.]), 0.1)


# ## lasso cyclic coordinate descent

# In[10]:

def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    weights = initial_weights
    max_weights_change = tolerance
    while(max_weights_change >= tolerance):
        max_weights_change = 0
        for i in range(len(weights)):
            old_weights_i = weights[i]
            weights[i] = lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)
            weights_change = abs(old_weights_i - weights[i])
            
            if weights_change > max_weights_change:
                max_weights_change = weights_change
    return weights   


# In[11]:

simple_features = ['sqft_living', 'bedrooms']
my_output = 'price'
l1_penalty = 1e7
tolerance = 1.0
initial_weights = np.zeros(3)


# In[134]:

simple_feature_matrix, output = convert_into_numpy_matrix(sales, simple_features, my_output)
normalized_simple_feature_matrix, normalizers = normalize_features(simple_features_matrix)
weights = lasso_cyclical_coordinate_descent(normalized_simple_feature_matrix, output, initial_weights, l1_penalty,tolerance)


# In[13]:

weights


# In[14]:

predictions = predict_output(normalized_simple_feature_matrix, weights)
diff = predictions-output
rss1 = sum(diff**2)
rss1


# ## lasso with more features

# In[190]:

all_features = ['bedrooms',
                'bathrooms',
                'sqft_living',
                'sqft_lot',
                'floors',
                'waterfront', 
                'view', 
                'condition', 
                'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built', 
                'yr_renovated']

my_l1_penalty1 = 1e7
my_l1_penalty2 = 1e8
my_initial_weights = np.zeros(len(all_features)+1)

(all_features_matrix, train_output) = convert_into_numpy_matrix(train, all_features, my_output)
normalized_all_features_matrix, train_norms = normalize_features(all_features_matrix)
weights1e7 = lasso_cyclical_coordinate_descent(normalized_all_features_matrix, train_output, my_initial_weights, 
                                               my_l1_penalty1, tolerance)


# In[194]:

normalized_weights1e7 = weights1e7/train_norms
normalized_weights1e7
my_df = pd.DataFrame(['const','bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot','waterfront','floors',
                'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated'])
my_df['1e7'] = pd.DataFrame(normalized_weights1e7)


# In[195]:

weights1e8 = lasso_cyclical_coordinate_descent(normalized_all_features_matrix, train_output, my_initial_weights,
                                              my_l1_penalty2, tolerance)
normalized_weights1e8 = weights1e8/train_norms
normalized_weights1e8
my_df['1e8'] = pd.DataFrame(normalized_weights1e8)


# In[196]:

my_l1_penalty3 = 1e4
my_tolerance = 5e5
weights1e4 = lasso_cyclical_coordinate_descent(normalized_all_features_matrix, train_output, my_initial_weights,
                                              my_l1_penalty3, my_tolerance)
normalized_weights1e4 = weights1e4/train_norms
normalized_weights1e4
my_df['1e4'] = pd.DataFrame(normalized_weights1e4)


# In[197]:

my_df


# ## calculating rss of each model on test data

# In[212]:

test_features_matrix, test_output = convert_into_numpy_matrix(test, all_features, 'price')

def calculate_rss(predictions):
    residuals = predictions - test_output
    rss = sum(predictions**2)
    return rss

predicitions1e7 = predict_output(test_features_matrix, normalized_weights1e7)
print "RSS 1e7 Model: ", calculate_rss(predicitions1e7), "\n"

predicitions1e8 = predict_output(test_features_matrix, normalized_weights1e8)
print "RSS 1e8 Model: ", calculate_rss(predicitions1e8), "\n"

predicitions1e4 = predict_output(test_features_matrix, normalized_weights1e4)
print "RSS 1e4 Model: ", calculate_rss(predicitions1e4), "\n"

normalized_weights1e4

