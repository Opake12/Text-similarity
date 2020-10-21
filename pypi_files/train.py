#!/usr/bin/env python
# coding: utf-8

# ### Model Training
#
# Now that you've created training and test data, you are ready to define and train a model.

import pandas as pd
import os
from sklearn.svm import LinearSVC


# should be the name of directory you created to save your features data
data_dir = 'plagiarism_data'
train_data = pd.read_csv(os.path.join(data_dir, "train.csv"), header=None, names=None)
test_data = pd.read_csv(os.path.join(data_dir, "test.csv"), header=None, names=None)

# Labels are in the first column
train_y = train_data.iloc[:,0]
train_x = train_data.iloc[:,1:]

# ### Define a model
model = LinearSVC()


# ### Train the model
model.fit(train_x, train_y)


# ---
# ## Evaluating Your Model
#
# Once your model is deployed, you can see how it performs when applied to our test data.
#
# The provided cell below, reads in the test data, assuming it is stored locally in `data_dir` and named `test.csv`. The labels and features are extracted from the `.csv` file.

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
import os

# read in test data, assuming it is stored locally
test_data = pd.read_csv("plagiarism_data/test.csv", header=None, names=None)

# labels are in the first column
test_y = test_data.iloc[:,0]
test_x = test_data.iloc[:,1:]


# ### Determine the accuracy of the model
# First: generate predicted, class labels
test_y_preds = model.predict(test_x)

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# test that your model generates the correct number of labels
assert len(test_y_preds)==len(test_y), 'Unexpected number of predictions.'
print('Test passed!')

# Second: calculate the test accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_y, test_y_preds)

print(accuracy)


## print out the array of predicted and true labels, if you want
print('\nPredicted class labels: ')
print(test_y_preds)
print('\nTrue class labels: ')
print(test_y.values)

# calculate and print the classification report
from sklearn.metrics import classification_report

print (classification_report(test_y, test_y_preds))



# ### Save Model

path = "./models/"

try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)


# save the model to disk
import pickle

filename = 'finalized_model.sav'
file = path+filename
pickle.dump(model, open(file, 'wb'))

# load the model from disk
# unmark if needed
# loaded_model = pickle.load(open(file, 'rb'))
