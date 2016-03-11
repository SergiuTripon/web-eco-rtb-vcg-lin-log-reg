
# local python files
import test
import load
import eval

# python packages
import random
import operator

########################################################################################################################


def run(reg_grad, learning_rate, threshold):

    data = load.load_file("input/spambase.data", load.data_format)

    def z_score(attribute):
        mu = sum(attribute) / float(len(attribute))
        sd = (sum([(x - mu) ** 2 for x in attribute]) / float(len(attribute) - 1)) ** 0.5
        zscore = [(x - mu) / float(sd) for x in attribute]
        return zscore

    def precondition(data_set):
        attributes = [email.attributes for email in data_set]
        features_zscore = [z_score(attribute) for attribute in attributes]

        for email, attributes in zip(data_set, features_zscore):
            email.attributes = attributes

    precondition(data)

    folds = [[], [], [], [], [], [], [], [], [], []]

    k = 0
    for email in data:
        folds[k].append(email)
        k = operator.mod((k + 1), 10)

    test_set = folds[0]

    train_set = []
    for fold in folds[1:]:
        train_set.extend(fold)

    random.shuffle(train_set)

    print('> Test set size:', len(test_set))
    print('> Train set size:', len(train_set), '\n')

    print('##########################################################################\n')

    # train
    trained_weights = test.train(train_set, reg_grad, learning_rate, threshold)

    # test
    results = test.test(trained_weights, test_set, reg_grad)

    # compute roc curve
    true_false_rates = eval.roc_curve(results, reg_grad, learning_rate)
    true_false_rates = true_false_rates[::-1]

    # compute auc
    eval.comp_auc(true_false_rates)

    print('##########################################################################')


########################################################################################################################
