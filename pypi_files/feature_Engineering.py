#!/usr/bin/env python
# coding: utf-8

# ## Read in the Data
##
# This data is a slightly modified version of a dataset created by Paul Clough (Information Studies) and Mark Stevenson (Computer Science), at the University of Sheffield. You can read all about the data collection and corpus, at [their university webpage](https://ir.shef.ac.uk/cloughie/resources/plagiarism_corpus.html).
#
# > **Citation for data**: Clough, P. and Stevenson, M. Developing A Corpus of Plagiarised Short Answers, Language Resources and Evaluation: Special Issue on Plagiarism and Authorship Analysis, In Press. [Download]


# import libraries
import pandas as pd
import numpy as np
import os


# This plagiarism dataset is made of multiple text files; each of these files has characteristics that are is summarized in a `.csv` file named `file_information.csv`, which we can read in using `pandas`.

csv_file = 'data/file_information.csv'
plagiarism_df = pd.read_csv(csv_file)

# print out the first few rows of data info
plagiarism_df.head()


# ## Types of Plagiarism
#
# Each text file is associated with one **Task** (task A-E) and one **Category** of plagiarism, which you can see in the above DataFrame.
#
# ###  Tasks, A-E
#
# Each text file contains an answer to one short question; these questions are labeled as tasks A-E. For example, Task A asks the question: "What is inheritance in object oriented programming?"
#
# ### Categories of plagiarism
#
# Each text file has an associated plagiarism label/category:
#
# **1. Plagiarized categories: `cut`, `light`, and `heavy`.**
# * These categories represent different levels of plagiarized answer texts. `cut` answers copy directly from a source text, `light` answers are based on the source text but include some light rephrasing, and `heavy` answers are based on the source text, but *heavily* rephrased (and will likely be the most challenging kind of plagiarism to detect).
#
# **2. Non-plagiarized category: `non`.**
# * `non` indicates that an answer is not plagiarized; the Wikipedia source text is not used to create this answer.
#
# **3. Special, source text category: `orig`.**
# * This is a specific category for the original, Wikipedia source text. We will use these files only for comparison purposes.

# ---
# ## Pre-Process the Data
#
# In the next few cells, you'll be tasked with creating a new DataFrame of desired information about all of the files in the `data/` directory. This will prepare the data for feature extraction and for training a binary, plagiarism classifier.
# * 4 columns: `File`, `Task`, `Category`, `Class`. The `File` and `Task` columns can remain unchanged from the original `.csv` file.
# * Convert all `Category` labels to numerical labels according to the following rules (a higher value indicates a higher degree of plagiarism):
#     * 0 = `non`
#     * 1 = `heavy`
#     * 2 = `light`
#     * 3 = `cut`
#     * -1 = `orig`, this is a special value that indicates an original file.
# * For the new `Class` column
#     * Any answer text that is not plagiarized (`non`) should have the class label `0`.
#     * Any plagiarized answer texts should have the class label `1`.
#     * And any `orig` texts will have a special label `-1`.
#
# ### Expected output
#
# After running your function, you should get a DataFrame with rows that looks like the following:
# ```
#
#         File	     Task  Category  Class
# 0	g0pA_taska.txt	a	  0   	0
# 1	g0pA_taskb.txt	b	  3   	1
# 2	g0pA_taskc.txt	c	  2   	1
# 3	g0pA_taskd.txt	d	  1   	1
# 4	g0pA_taske.txt	e	  0	   0
# ...
# ...
# 99   orig_taske.txt    e     -1      -1
#
# ```

# Read in a csv file and return a transformed dataframe
def numerical_dataframe(csv_file='data/file_information.csv'):
    '''Reads in a csv file which is assumed to have `File`, `Category` and `Task` columns.
       This function does two things:
       1) converts `Category` column values to numerical values
       2) Adds a new, numerical `Class` label column.
       The `Class` column will label plagiarized answers as 1 and non-plagiarized as 0.
       Source texts have a special label, -1.
       :param csv_file: The directory for the file_information.csv file
       :return: A dataframe with numerical categories and a new `Class` label column'''


    df = pd.read_csv(csv_file)

    # create `Class` column
    df['Class'] = df['Category'].map({
        'non': 0,
        'heavy': 1,
        'light': 1,
        'cut': 1,
        'orig': -1
    })

    # encode Categories
    df['Category'] = df['Category'].map({
        'non': 0,
        'heavy': 1,
        'light': 2,
        'cut': 3,
        'orig': -1
    })

    return df


# ### Test cells
# informal testing, print out the results of a called function
# create new `transformed_df`
transformed_df = numerical_dataframe(csv_file ='data/file_information.csv')

# check work
# check that all categories of plagiarism have a class label = 1
transformed_df.head(10)



# test cell that creates `transformed_df`, if tests are passed
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

# importing tests
import problem_unittests as tests

# test numerical_dataframe function
tests.test_numerical_df(numerical_dataframe)

# if above test is passed, create NEW `transformed_df`
transformed_df = numerical_dataframe(csv_file ='data/file_information.csv')

# check work
print('\nExample data: ')
transformed_df.head()


# ## Text Processing & Splitting Data
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
import helpers

# create a text column
text_df = helpers.create_text_column(transformed_df)
text_df.head()

# after running the cell above
# check out the processed text for a single file, by row index
row_idx = 1 # feel free to change this index

sample_text = text_df.iloc[0]['Text']

print('Sample processed text:\n\n', sample_text)


# Split data into training and test sets
random_seed = 1 # can change; set for reproducibility

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
import helpers

# create new df with Datatype (train, test, orig) column
# pass in `text_df` from above to create a complete dataframe, with all the information you need
complete_df = helpers.train_test_dataframe(text_df, random_seed=random_seed)

# check results
complete_df.head(10)


from sklearn.feature_extraction.text import CountVectorizer

# Calculate the ngram containment for one answer file/source file pair in a df
def calculate_containment(df, n, answer_filename):
    '''Calculates the containment between a given answer text and its associated source text.
       This function creates a count of ngrams (of a size, n) for each text file in our data.
       Then calculates the containment by finding the ngram count for a given answer text,
       and its associated source text, and calculating the normalized intersection of those counts.
       :param df: A dataframe with columns,
           'File', 'Task', 'Category', 'Class', 'Text', and 'Datatype'
       :param n: An integer that defines the ngram size
       :param answer_filename: A filename for an answer text in the df, ex. 'g0pB_taskd.txt'
       :return: A single containment value that represents the similarity
           between an answer text and its source text.
    '''

    a_text = df.loc[df['File'] == answer_filename, 'Text'].values[0]
    a_task = df.loc[df['File'] == answer_filename, 'Task'].values[0]

    s_text = df.loc[(df['Category'] == -1) & (df['Task'] == a_task), 'Text'].values[0]

    # instantiate an ngram counter
    counts = CountVectorizer(analyzer='word', ngram_range=(n,n))
    # create array of n-gram counts for the answer and source text
    ngram_array = counts.fit_transform([a_text, s_text]).toarray()

    # Calculate containment
    # sum up number of the intersection counts
    # count up the number of n-grams in the answer text
    # normalize and get final containment value
    answer_idx = 0
    intersection_list = np.amin(ngram_array, axis=0)
    containment = np.sum(intersection_list) / np.sum(ngram_array[answer_idx])

    return containment



# Compute the normalized LCS given an answer text and a source text
def lcs_norm_word(answer_text, source_text):
    '''Computes the longest common subsequence of words in two texts; returns a normalized value.
       :param answer_text: The pre-processed text for an answer text
       :param source_text: The pre-processed text for an answer's associated source text
       :return: A normalized LCS value'''

    # split answer and source text into list of words
    a_words = answer_text.split()
    s_words = source_text.split()

    # calculate the length of answer and source text
    a_len = len(a_words)
    s_len = len(s_words)

    # create LCS matrix and populate it with zeros and +1 additional row and column
    lcs_matrix = np.zeros((a_len+1, s_len+1), dtype=int)

    # iterate through words
    # source: https://www.geeksforgeeks.org/python-program-for-longest-common-subsequence/

    for a in range(a_len+1):
        for s in range(s_len+1):
            if a==0 or s==0:
                lcs_matrix[a][s]=0 # if at index zero
            elif a_words[a-1] == s_words[s-1]:
                lcs_matrix[a][s] = lcs_matrix[a-1][s-1] + 1 # if match is found increment by 1
            else:
                lcs_matrix[a][s] = max(lcs_matrix[a-1][s], lcs_matrix[a][s-1])

    return int(lcs_matrix[a_len][s_len]) / a_len


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# Function returns a list of containment features, calculated for a given n
# Should return a list of length 100 for all files in a complete_df
def create_containment_features(df, n, column_name=None):

    containment_values = []

    if(column_name==None):
        column_name = 'c_'+str(n) # c_1, c_2, .. c_n

    # iterates through dataframe rows
    for i in df.index:
        file = df.loc[i, 'File']
        # Computes features using calculate_containment function
        if df.loc[i,'Category'] > -1:
            c = calculate_containment(df, n, file)
            containment_values.append(c)
        # Sets value to -1 for original tasks
        else:
            containment_values.append(-1)

    print(str(n)+'-gram containment features created!')
    return containment_values


# ### Creating LCS features
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# Function creates lcs feature and add it to the dataframe
def create_lcs_features(df, column_name='lcs_word'):

    lcs_values = []

    # iterate through files in dataframe
    for i in df.index:
        # Computes LCS_norm words feature using function above for answer tasks
        if df.loc[i,'Category'] > -1:
            # get texts to compare
            answer_text = df.loc[i, 'Text']
            task = df.loc[i, 'Task']
            # we know that source texts have Class = -1
            orig_rows = df[(df['Class'] == -1)]
            orig_row = orig_rows[(orig_rows['Task'] == task)]
            source_text = orig_row['Text'].values[0]

            # calculate lcs
            lcs = lcs_norm_word(answer_text, source_text)
            lcs_values.append(lcs)
        # Sets to -1 for original tasks
        else:
            lcs_values.append(-1)

    print('LCS features created!')
    return lcs_values


# Create a features DataFrame by selecting an `ngram_range`
# Define an ngram range
ngram_range = range(1,19)

# The following code may take a minute to run, depending on your ngram_range
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
features_list = []

# Create features in a features_df
all_features = np.zeros((len(ngram_range)+1, len(complete_df)))

# Calculate features for containment for ngrams in range
i=0
for n in ngram_range:
    column_name = 'c_'+str(n)
    features_list.append(column_name)
    # create containment features
    all_features[i]=np.squeeze(create_containment_features(complete_df, n))
    i+=1

# Calculate features for LCS_Norm Words
features_list.append('lcs_word')
all_features[i]= np.squeeze(create_lcs_features(complete_df))

# create a features dataframe
features_df = pd.DataFrame(np.transpose(all_features), columns=features_list)

# Print all features/columns
print()
print('Features: ', features_list)
print()


# print some results
features_df.head(10)


# ## Correlated Features
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# Create correlation matrix for just Features to determine different models to test
corr_matrix = features_df.corr().abs().round(2)


import matplotlib.pyplot as plt
#import seaborn as sb


#print(sb.heatmap(corr_matrix, cmap='viridis',linecolor='white',linewidths=1))


# Create selected train/test data
# Takes in dataframes and a list of selected features (column names)
# and returns (train_x, train_y), (test_x, test_y)
def train_test_data(complete_df, features_df, selected_features):
    '''Gets selected training and test features from given dataframes, and
       returns tuples for training and test features and their corresponding class labels.
       :param complete_df: A dataframe with all of our processed text data, datatypes, and labels
       :param features_df: A dataframe of all computed, similarity features
       :param selected_features: An array of selected features that correspond to certain columns in `features_df`
       :return: training and test features and labels: (train_x, train_y), (test_x, test_y)'''
    # concat dataframes
    df = pd.concat([complete_df, features_df[selected_features]], axis=1)

    # get the training features
    train_x = df.query("Datatype == 'train'")[selected_features].values
    # And training class labels (0 or 1)
    train_y = df.query("Datatype == 'train'")['Class'].values

    # get the test features and labels
    test_x = df.query("Datatype == 'test'")[selected_features].values
    test_y = df.query("Datatype == 'test'")['Class'].values

    return (train_x, train_y), (test_x, test_y)



# ### Test cells
#
# Below, test out your implementation and create the final train/test data.

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
test_selection = list(features_df)[:2] # first couple columns as a test
# test that the correct train/test data is created
(train_x, train_y), (test_x, test_y) = train_test_data(complete_df, features_df, test_selection)

# params: generated train/test data
tests.test_data_split(train_x, train_y, test_x, test_y)


# Select your list of features, this should be column names from features_df
# ex. ['c_1', 'lcs_word']
selected_features = ['c_1','c_2', 'c_9', 'lcs_word']


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

(train_x, train_y), (test_x, test_y) = train_test_data(complete_df, features_df, selected_features)

# check that division of samples seems correct
# these should add up to 95 (100 - 5 original files)
print('Training size: ', len(train_x))
print('Test size: ', len(test_x))
print()
print('Training df sample: \n', train_x[:10])


# ---
# ## Creating Final Data Files



def make_csv(x, y, filename, data_dir):
    '''Merges features and labels and converts them into one csv file with labels in the first column.
       :param x: Data features
       :param y: Data labels
       :param file_name: Name of csv file, ex. 'train.csv'
       :param data_dir: The directory where files will be saved
       '''
    # make data dir, if it does not exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)


    pd.concat([pd.DataFrame(y), pd.DataFrame(x)], axis=1).to_csv(os.path.join(data_dir, filename),
                                                                             header=False, index=False)

    # nothing is returned, but a print statement indicates that the function has run
    print('Path created: '+str(data_dir)+'/'+str(filename))


# can change directory, if you want
data_dir = 'plagiarism_data'

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

make_csv(train_x, train_y, filename='train.csv', data_dir=data_dir)
make_csv(test_x, test_y, filename='test.csv', data_dir=data_dir)
