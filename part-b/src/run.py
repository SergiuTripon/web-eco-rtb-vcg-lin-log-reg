
# local python files
import test
import load

# python packages
import random
import operator

########################################################################################################################


def run(reg_model, reg_grad, grad_func, learning_rate, threshold):

    data = load.load_file("input/spambase.data", load.data_format)

    def z_score(feature):
        mu = sum(feature) / float(len(feature))
        sd = (sum([(x - mu) ** 2 for x in feature]) / float(len(feature) - 1)) ** 0.5
        zscore = [(x - mu) / float(sd) for x in feature]
        return zscore

    def precondition(data_set):
        features = [email.features for email in data_set]
        features_zscore = [z_score(feature) for feature in features]

        for email, features in zip(data_set, features_zscore):
            email.features = features

    precondition(data)

    folds = []
    for i in range(10):
        folds.append([])

    k = 0
    for data_point in data:
        folds[k].append(data_point)
        k = operator.mod((k + 1), 10)

    test_set = folds[0]

    train_set = []
    for fold in folds[1:]:
        train_set.extend(fold)

    random.shuffle(train_set)

    print('> Testing count:', len(test_set))
    print('> Training count:', len(train_set), '\n')

    print('##########################################################################\n')
    print('> Running', grad_func.__name__, '\n')
    test.test(test_set, train_set, reg_model, reg_grad, grad_func, learning_rate, threshold)


########################################################################################################################
