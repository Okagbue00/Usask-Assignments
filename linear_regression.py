# OKAGBUE ONYEKA FRANCIS
# CMPT 317
# MARCH 30TH 2023
# 11279373


def predict(point, weights):
    """ predicts the output value for the given point, using
    the given set of weights
    
    :params:
    point: record with keys "features" and "output"
    weights: list of weights for linear regression, one
    for each feature
    
    returns: integer. the predicted output value for the point """
    total = 0
    for i in range(len(weights)):
        total += weights[i] * point["features"][i]
        
    return total


def train(data, numsteps, alpha):
    """ trains a list of weights for linear regression
    
    :params:
    data: list of data points.  Each point is a record with
    fields "features" and "output"
    numsteps: integer. Number of training steps to use
    alpha: float. learning rate for the regression update rule
    
    returns: a list of weights, one for each feature
    """
    # initialize the weights to arbitrary values
    firstpoint = data[0]
    numfeatures = len(firstpoint["features"])
    w = [0] * numfeatures
        
    #TODO: train the weights here

    # would perform the steps
    for m in range(numsteps):
        gradient = []
        for _ in range(numfeatures):
            gradient.append(0)

        for values in data:
            pred = predict(values, w)

            # would check for each feature
            for y in range(numfeatures):
                error = pred - values["output"]
                scaled_error = 2 * error / len(data)
                feature_contribution = scaled_error * values["features"][y]
                gradient[y] += feature_contribution

        for y in range(numfeatures):
            w[y] -= alpha * gradient[y]

    return w
    
    
def total_error(data, weights):
    """ computes and returns the total mean squared error
    for all points in the given data set
    
    :params:
    data: list of data points, each point is a record with keys "features" and "output"
    weights: list of weights for linear regression, one weight
    per feature in the data
    
    return: float. the total mean squared error """
    total = 0
    for point in data:
        prediction = predict(point, weights)
        # a bit icky to use int, but otherwise the errors can
        # get too big for a Python float
        total += int((point["output"] - prediction)**2)
        
    total = total//len(data)
    return total