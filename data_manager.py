import random as random

def read_data(filename):
    """ returns a list of data points read from a file name.
    Each data point is stored in a dictionary with fields
    "features" and "output". For mathematical convenience,
    a dummy feature with value 1 is added to all data points 
    to be the 'feature' for w0 (the bias).  Data is assumed to be integers
    
    :params:
    filename: name of the data file to read
    
    returns: a list of records """
    f = open(filename, "r")
    data = []
    for line in f:
        line = line.rstrip().split()
        line = [int(x) for x in line]
        point = {}
        point["features"] = tuple([1] + line[0:-1])
        point["output"] = line[-1]
        data.append(point)    
    
    f.close()
    return data


def partition_data(data, x=5):
    """ partitions a set of data into training and test sets
    returns a dictionary-of-records.  The keys of the dictionary are the index of the test set.  Each record has two fields, the strings "train" and "test"

    :params:
    data: a list of tuples, each tuple is a data point
    x: is a integer for the data to create

    returns: a dictionary of train/test set pairs """

    def split_data(data, start, end):
        mid = (start + end) // 2

        return data[start: mid], data[mid: end]

    data_par = list(data)
    random.shuffle(data_par)

    fold_size = len(data) // x
    partitions = {}

    for i in range(x):
        start_index = fold_size * i
        end_index = fold_size * (i + 1) if i < x - 1 else len(data_par)

        train_data, test_data=split_data(data_par, start_index, end_index)
        partitions[i + 1] = {"train": train_data, "test": test_data}

    return partitions


