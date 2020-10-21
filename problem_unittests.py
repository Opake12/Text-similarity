from unittest.mock import MagicMock, patch
import sklearn.naive_bayes
import numpy as np
import pandas as pd
import re

# test csv file
TEST_CSV = 'data/test_info.csv'

class AssertTest(object):
    '''Defines general test behavior.'''
    def __init__(self, params):
        self.assert_param_message = '\n'.join([str(k) + ': ' + str(v) + '' for k, v in params.items()])

    def test(self, assert_condition, assert_message):
        assert assert_condition, assert_message + '\n\nUnit Test Function Parameters\n' + self.assert_param_message

def _print_success_message():
    print('Tests Passed!')

# test clean_dataframe
def test_numerical_df(numerical_dataframe):

    # test result
    transformed_df = numerical_dataframe(TEST_CSV)

    # Check type is a DataFrame
    assert isinstance(transformed_df, pd.DataFrame), 'Returned type is {}.'.format(type(transformed_df))

    # check columns
    column_names = list(transformed_df)
    assert 'File' in column_names, 'No File column, found.'
    assert 'Task' in column_names, 'No Task column, found.'
    assert 'Category' in column_names, 'No Category column, found.'
    assert 'Class' in column_names, 'No Class column, found.'

    # check conversion values
    assert transformed_df.loc[0, 'Category'] == 1, '`heavy` plagiarism mapping test, failed.'
    assert transformed_df.loc[2, 'Category'] == 0, '`non` plagiarism mapping test, failed.'
    assert transformed_df.loc[30, 'Category'] == 3, '`cut` plagiarism mapping test, failed.'
    assert transformed_df.loc[5, 'Category'] == 2, '`light` plagiarism mapping test, failed.'
    assert transformed_df.loc[37, 'Category'] == -1, 'original file mapping test, failed; should have a Category = -1.'
    assert transformed_df.loc[41, 'Category'] == -1, 'original file mapping test, failed; should have a Category = -1.'

    _print_success_message()


def test_data_split(train_x, train_y, test_x, test_y):

    # check types
    assert isinstance(train_x, np.ndarray),\
        'train_x is not an array, instead got type: {}'.format(type(train_x))
    assert isinstance(train_y, np.ndarray),\
        'train_y is not an array, instead got type: {}'.format(type(train_y))
    assert isinstance(test_x, np.ndarray),\
        'test_x is not an array, instead got type: {}'.format(type(test_x))
    assert isinstance(test_y, np.ndarray),\
        'test_y is not an array, instead got type: {}'.format(type(test_y))

    # should hold all 95 submission files
    assert len(train_x) + len(test_x) == 95, \
        'Unexpected amount of train + test data. Expecting 95 answer text files, got ' +str(len(train_x) + len(test_x))
    assert len(test_x) > 1, \
        'Unexpected amount of test data. There should be multiple test files.'

    # check shape
    assert train_x.shape[1]==2, \
        'train_x should have as many columns as selected features, got: {}'.format(train_x.shape[1])
    assert len(train_y.shape)==1, \
        'train_y should be a 1D array, got shape: {}'.format(train_y.shape)

    _print_success_message()
