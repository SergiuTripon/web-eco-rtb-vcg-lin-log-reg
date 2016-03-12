
# local python files
import load
import test
import eval

# python packages
import random
import operator

########################################################################################################################


# starts the whole workflow
# 1. loads the data
# 2. preconditions the data
# 3. partitions the data
# 4. creates test set
# 5. creates train set
# 6. randomizes train set
# 7. runs train
# 8. runs test
# 9. runs roc curve
# 10. runs auc
def run(reg_grad, learning_rate, threshold):

    # load data
    data = load.load_file("input/spambase.data", load.data_format)

    # computes z-score for an attribute
    def z_score(attribute):
        # compute mean
        mu = sum(attribute) / float(len(attribute))
        # compute variance and then standard deviations
        sd = (sum([(x - mu) ** 2 for x in attribute]) / float(len(attribute) - 1)) ** 0.5
        # compute z-score
        zscore = [(x - mu) / float(sd) for x in attribute]
        # return zscore
        return zscore

    # preconditions the data set using z-score
    def precondition(data_set):
        # get attributes for all email in data set
        attributes = [email.attributes for email in data_set]
        # for each attribute, calculate its z-score
        attributes_zscore = [z_score(attribute) for attribute in attributes]

        # for each email in the data set and attribute in z-score preconditioned attributes
        for email, attributes in zip(data_set, attributes_zscore):
            # update each email's attributes with its z-score preconditioned attributes
            email.attributes = attributes

    # precondition the data
    precondition(data)

    # 10 lists within one list to hold 10 folds within the folds list
    folds = [[], [], [], [], [], [], [], [], [], []]

    # set k to 0
    k = 0
    # for each email in the data
    for email in data:
        # add email to the specific fold specified by k
        # first email will be added to fold 0
        folds[k].append(email)
        # update k as (k + 1) mod 10
        # e.g. 0 + 1 mod 10 = 1 mod 10 = 1
        # e.g. 1 + 1 mod 10 = 2 mod 10 = 2
        # e.g. 9 + 1 mod 10 = 10 mod 10 = 0
        # therefore, emails will be distributed equally between folds
        # with the exception of the first email which was added to fold 0 automatically
        # therefore fold 0 consists of 461 emails
        # the other folds consist of 460 emails each
        k = operator.mod((k + 1), 10)

    # define test set as the first fold
    test_set = folds[0]

    # define train set as all the other folds except the first one
    # as the first one will be used as the test set
    train_set = []
    for fold in folds[1:]:
        train_set.extend(fold)

    # randomize the elements in the training set
    random.shuffle(train_set)

    # do some printing to show progress
    print('> Test set size:', len(test_set))
    print('> Train set size:', len(train_set), '\n')
    print('##########################################################################\n')

    # train and return the output in the trained_weights variable
    trained_weights = test.train(train_set, reg_grad, learning_rate, threshold)

    # test and return the output in the results variable
    results = test.test(trained_weights, test_set, reg_grad)

    # compute roc curve
    true_false_rates = eval.roc_curve(results, reg_grad, learning_rate)
    # reverse list holding the true false rates before sending it to auc function
    true_false_rates = true_false_rates[::-1]

    # compute auc
    eval.comp_auc(true_false_rates)

    # do some printing to show progress
    print('##########################################################################')


########################################################################################################################
