
# python package
from math import e

########################################################################################################################


def log_sigmoid(x):
    return 1 / (1 + (e ** -x))


########################################################################################################################


def lin_reg_model(weights, features):
    return sum([x * y for x, y in zip(weights, features)])


def lin_reg_grad(reg_model, feature, label):
    return (reg_model - label) * feature


########################################################################################################################


def log_reg_model(weights, features):
    test1 = log_sigmoid(lin_reg_model(weights, features))
    return test1


def log_reg_grad(reg_model, feature, label):
    test2 = reg_model * (1 - reg_model) * lin_reg_grad(reg_model, feature, label)
    return test2


########################################################################################################################


def lin_sgd(weights, train_set, learning_rate):

    res3 = []
    for email in train_set:
        res1 = sum([x * y for x, y in zip(weights, email.features)])
        res2 = [(res1 - email.label) * f for f in email.features]
        res3 = [w - learning_rate * x for w, x in zip(weights, res2)]
        weights = res3
    return res3

########################################################################################################################


def lin_bgd(weights, train_set, learning_rate):

    res4 = []
    for i in range(len(weights)):
        res3 = []
        for email in train_set:
            res1 = sum([x * y for x, y in zip(weights, email.features)])
            res2 = (res1 - email.label) * email.features[i]
            res3 += [res2]
        res4 += [sum(res3)]

    res5 = [w - learning_rate * x for w, x in zip(weights, res4)]
    return res5


########################################################################################################################


def log_sgd(weights, train_set, learning_rate):

    res4 = []
    for email in train_set:
        res1 = log_sigmoid(sum([x * y for x, y in zip(weights, email.features)]))
        res2 = [(res1 - email.label) * f for f in email.features]
        res3 = [res1 * (1 - res1) * x for x in res2]
        res4 = [x - learning_rate * y for x, y in zip(weights, res3)]
        weights = res4
    return res4


########################################################################################################################

def log_bgd(weights, train_set, learning_rate):

    res5 = []
    for i in range(len(weights)):
        res4 = []
        for email in train_set:
            res1 = log_sigmoid(sum([x * y for x, y in zip(weights, email.features)]))
            res2 = (res1 - email.label) * email.features[i]
            res3 = res1 * (1 - res1) * res2
            res4 += [res3]
        res5 += [sum(res4)]

    res6 = [w - learning_rate * x for w, x in zip(weights, res5)]
    return res6


########################################################################################################################
