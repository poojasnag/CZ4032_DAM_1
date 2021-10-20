import csv
from typing import List

def read_data(path: str) -> List:
    """
    Read dataset and convert into a list.

    :param path: directory of *.data file.
    """
    data = []
    with open(path, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for line in reader:
            data.append(line)
        while [] in data:
            data.remove([])
    return data


def read_scheme(path: str):
    """
    Read scheme file *.names and write down attributes and value types.

    :param path: directory of *.names file.
    """
    with open(path, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        attributes = next(reader)
        value_type = next(reader)
    return attributes, value_type


def str2numerical(data, value_type):
    """
    convert string-type value into float-type.

    :param data: data list returned by read_data.
    :param value_type: list returned by read_scheme.
    """
    size = len(data)
    columns = len(data[0])
    for i in range(size):
        for j in range(columns-1):
            if value_type[j] == 'numerical' and data[i][j] != '?':
                data[i][j] = float(data[i][j])
    return data


def read(data_path, scheme_path):
    """
    Main method in this file, to get data list after processing and scheme list.

    :param data_path: tell where *.data file stores.
    :param scheme_path: tell where *.names file stores.
    """
    data = read_data(data_path)
    attributes, value_type = read_scheme(scheme_path)
    data = str2numerical(data, value_type)
    return data, attributes, value_type
