from typing import List
import numpy as np
import pandas as pd
from utils.dataset import Dataset

def get_mode(arr: List):
    """
    Identify the mode of a list, both effective for numerical and categorical list. When there exists too many modes having the same frequency, return the first one.
    """
    arr_appear = dict((a, arr.count(a)) for a in arr)  # count appearance times of each key
    if max(arr_appear.values()) == 1:  # if max time is 1
        return  # no mode here
    else:
        for k, v in arr_appear.items():  # else, mode is the number which has max time
            if v == max(arr_appear.values()):
                return k  # return first number if has many modes


def fill_missing_values(data: List, column_no) -> List:
    """
    Fill missing values in column column_no, when missing values ration below 50%.
    
    :param data: Original data list
    :param column_no: Identify the column number to be filled
    """
    size = len(data)
    column_data = [x[column_no] for x in data]  # get that column
    while '?' in column_data:
        column_data.remove('?')
    mode = get_mode(column_data)
    for i in range(size):
        if data[i][column_no] == '?':
            data[i][column_no] = mode  # fill in mode
    return data


def get_discretization_data(data_column) -> List:
    size = len(data_column)
    result_list = []
    for i in range(size):
        result_list.append(data_column[i])
    return result_list


def replace_numerical(data, list_data, column_no) -> None:
    """
    Discretization method using equal width binning (3 bins set based on empirical experiments)
    """
    size = len(data)
    new_list = list(pd.cut(list_data, 3, labels=False))
    for i in range(size):
        data[i][column_no] = new_list[i]


def replace_categorical(data, column_no):
    """
    Replace categorical values with a positive integer.

    :param data: Original data list
    :param column_no: identify which column to be processed
    """
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


def discard(data: List, discard_list: List) -> List:
    """
    Discard all the column with its column_no in discard_list

    :param data: Original data list
    :param column_no: List of column numbers to be discarded
    """
    size = len(data)
    length = len(data[0])
    data_result = []
    for i in range(size):
        data_result.append([])
        for j in range(length):
            if j not in discard_list:
                data_result[i].append(data[i][j])
    return data_result


def pre_process(data: List, attribute: List, value_type: List) -> Dataset:
    """
    Main preprocessing logic.

    :param data: original data list
    :param attribute: a list of the name of attribute
    :param value_type: a list identifying the type of each column
    :return: a data table after process
    """
    column_num = len(data[0])
    size = len(data)
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


def get_minsup(label, minsup_dict: dict) -> float:
    """
    Get unique minsup value for that class label
    """
    minsup_value = minsup_dict.get(label)
    return minsup_value
