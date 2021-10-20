import numpy as np
import pandas as pd
from utils.dataset import Dataset

# Identify the mode of a list, both effective for numerical and categorical list. When there exists too many modes
#   having the same frequency, return the first one.
# arr: a list need to find mode
def get_mode(arr):
    # print(arr)
    arr_appear = dict((a, arr.count(a)) for a in arr)   # count appearance times of each key
    if max(arr_appear.values()) == 1:       # if max time is 1
        return      # no mode here
    else:
        for k, v in arr_appear.items():     # else, mode is the number which has max time
            if v == max(arr_appear.values()):
                return k  # return first number if has many modes


# Fill missing values in column column_no, when missing values ration below 50%.
# data: original data list
# column_no: identify the column No. of that to be filled
def fill_missing_values(data, column_no):
    size = len(data)
    column_data = [x[column_no] for x in data]      # get that column
    while '?' in column_data:
        column_data.remove('?')
    mode = get_mode(column_data)
    for i in range(size):
        if data[i][column_no] == '?':
            data[i][column_no] = mode              # fill in mode
    return data

def get_discretization_data(data_column):
    size = len(data_column)
    result_list = []
    for i in range(size):
        result_list.append(data_column[i])
    return result_list

def replace_numerical(data, list_data, column_no):
    size = len(data)
    new_list = list(pd.cut(list_data, 3, labels=False))
    for i in range(size):
        data[i][column_no] = new_list[i]


# Replace categorical values with a positive integer.
# data: original data table
# column_no: identify which column to be processed
def replace_categorical(data, column_no):
    size = len(data)
    classes = set([x[column_no] for x in data])
    classes_no = dict([(label, 0) for label in classes])
    j = 1
    for i in classes:
        classes_no[i] = j
        j += 1
    for i in range(size):
        data[i][column_no] = classes_no[data[i][column_no]]
    return data, classes_no


# Discard all the column with its column_no in discard_list
# data: original data set
# discard_list: a list of column No. of the columns to be discarded
def discard(data, discard_list):
    size = len(data)
    length = len(data[0])
    data_result = []
    for i in range(size):
        data_result.append([])
        for j in range(length):
            if j not in discard_list:
                data_result[i].append(data[i][j])
    return data_result


# Main method here, see Description in detail
# data: original data table
# attribute: a list of the name of attribute
# value_type: a list identifying the type of each column
# Returned value: a data table after process
def pre_process(data, attribute, value_type):
    column_num = len(data[0])
    size = len(data)
    class_column = [x[-1] for x in data]
    discard_list = []
    for i in range(0, column_num - 1):
        data_column = [x[i] for x in data]
        # process missing values
        missing_values_ratio = data_column.count('?') / size
        if missing_values_ratio > 0.5:
            discard_list.append(i)
            continue
        elif missing_values_ratio > 0:
            data = fill_missing_values(data, i)
            data_column = [x[i] for x in data]

        # discretization
        if value_type[i] == 'numerical':
            list_data = get_discretization_data(data_column)
            replace_numerical(data,list_data,i)

        elif value_type[i] == 'categorical':
        # else:
            data, classes_no = replace_categorical(data, i)
            print(attribute[i] + ":", classes_no)   # print out replacement list
    # discard
    if len(discard_list) > 0:
        data = discard(data, discard_list)
        print("discard:", discard_list)             # print out discard list


    data = Dataset(data, value_type, attribute)
    return data

# multiple minsup
# get unique minsup value for that class label
def get_minsup(label, minsup_dict):
    minsup_value = minsup_dict.get(label)
    return minsup_value
