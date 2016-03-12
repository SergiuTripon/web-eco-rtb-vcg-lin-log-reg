
# python package
from math import e

########################################################################################################################


# computes logistic sigmoid
def log_sigmoid(x):
    log_sig = 1 / (1 + (e ** -x))
    return log_sig


########################################################################################################################


# computes linear regression with stochastic gradient descent
def lin_sgd(weights, train_set, learning_rate):

    # list to hold sgd
    sgd = []
    # for every email in train set
    for email in train_set:
        # compute updated weights
        updated_weights = sum([x * y for x, y in zip(weights, email.attributes)])
        # compute gradient
        gradient = [(updated_weights - email.gold) * f for f in email.attributes]
        # compute sgd
        sgd = [w - learning_rate * g for w, g in zip(weights, gradient)]
        # update weights as sgd
        weights = sgd
    # return sgd
    return sgd

########################################################################################################################


# computes linear regression with batch gradient descent
def lin_bgd(weights, train_set, learning_rate):

    # list to hold gradients' sum
    gradients_sum = []
    # for loop running 57 times, the number of weights
    for i in range(len(weights)):
        # list to hold gradients
        gradients = []
        # for every email in train set
        for email in train_set:
            # compute updated weights
            updated_weights = sum([x * y for x, y in zip(weights, email.attributes)])
            # compute gradient
            gradient = (updated_weights - email.gold) * email.attributes[i]
            # add every gradient to gradients list
            gradients += [gradient]
        # compute sum of gradients list and add it to gradients_sum list
        gradients_sum += [sum(gradients)]
    # compute bgd
    bgd = [w - learning_rate * g for w, g in zip(weights, gradients_sum)]
    # return bgd
    return bgd


########################################################################################################################


# computes logistic regression with stochastic gradient descent
def log_sgd(weights, train_set, learning_rate):

    # list to hold sgd
    sgd = []
    # for every email in the train set
    for email in train_set:
        # compute updated weights using logistic sigmoid
        updated_weights = log_sigmoid(sum([x * y for x, y in zip(weights, email.attributes)]))
        # compute linear gradient
        lin_grad = [(updated_weights - email.gold) * f for f in email.attributes]
        # compute logistic gradient
        log_grad = [updated_weights * (1 - updated_weights) * x for x in lin_grad]
        # compute sgd
        sgd = [x - learning_rate * g for x, g in zip(weights, log_grad)]
        # update weights as sgd
        weights = sgd
    # return sgd
    return sgd


########################################################################################################################


# computes logistic regression with batch gradient descent
def log_bgd(weights, train_set, learning_rate):

    # list to hold gradients' sum
    gradients_sum = []
    # for loop running 57 times, the number of weights
    for i in range(len(weights)):
        # list to hold gradients
        gradients = []
        # for every email in train set
        for email in train_set:
            # compute updated weights using logistic sigmoid
            updated_weights = log_sigmoid(sum([x * y for x, y in zip(weights, email.attributes)]))
            # compute linear gradient
            lin_grad = (updated_weights - email.gold) * email.attributes[i]
            # compute logistic gradient
            log_grad = updated_weights * (1 - updated_weights) * lin_grad
            # add every gradient to gradients list
            gradients += [log_grad]
        # compute sum of gradients list and add it to gradients_sum list
        gradients_sum += [sum(gradients)]
    # compute bgd
    bgd = [w - learning_rate * g for w, g in zip(weights, gradients_sum)]
    # return bgd
    return bgd


########################################################################################################################
