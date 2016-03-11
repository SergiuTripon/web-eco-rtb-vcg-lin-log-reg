
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

    print('> Test set size:', len(test_set))
    print('> Train set size:', len(train_set), '\n')

    print('##########################################################################\n')
    print('> Training', reg_grad.__name__, '\n')

    # train
    init_weights = 57 * [0.0]
    trained_weights = test.train(init_weights, train_set, reg_grad, learning_rate, threshold)

    # test
    results = test.test(trained_weights, test_set, reg_grad)

    # compute roc curve
    true_false_rates = eval.roc_curve(results)
    true_false_rates = true_false_rates[::-1]

    # write roc data to file
    with open('output/text/{}_{}_roc.txt'.format(reg_grad.__name__, learning_rate).lower(), mode='w') as fd:
        for true_pos_rate, false_pos_rate in true_false_rates:
            fd.write('{}, {}\n'.format(true_pos_rate, false_pos_rate))
    print('> Saved ROC data to file', '\n')

    # compute auc
    auc = eval.comp_auc(true_false_rates)
    print('> AUC:', auc, '\n')

    print('##########################################################################')


########################################################################################################################
