
# python package
import operator

########################################################################################################################


# Result object
class Result(object):

    def __init__(self, trained_weight, gold):

        # result weight
        self.weight = trained_weight

        # result gold
        self.gold = gold


########################################################################################################################

# computes roc curve data
def roc_curve(results, reg_grad, learning_rate):

    # sort results by weight attribute
    results = sorted(results, key=operator.attrgetter("weight"))

    # list to hold matching true and false positive rates
    true_false_rates = []

    # selects one result at a time
    # compares its weight with the weight of all the other results
    # for every result in results
    for result in results:

        # set rates start value to 0
        true_pos = [0.0]
        true_neg = [0.0]
        false_neg = [0.0]
        false_pos = [0.0]

        # for every result1 in results
        for result1 in results:
            # if result1's gold is 1 and result1's weight is larger or equal to result's weight
            if result1.gold == 1 and result1.weight >= result.weight:
                # increment true positive rate
                true_pos[0] += 1.0
            # if result1's gold is 0 and result1's weight is smaller than result's weight
            elif result1.gold == 0 and result1.weight < result.weight:
                # increment true negative rate
                true_neg[0] += 1.0
            # if result1's gold is 0 and result1's weight is larger or equal to result's weight
            elif result1.gold == 0 and result1.weight >= result.weight:
                # increment false positive rate
                false_pos[0] += 1.0
            # if result1's gold is 1 and result1's weight is smaller than result's weight
            elif result1.gold == 1 and result1.weight < result.weight:
                # increment false negative rate
                false_neg[0] += 1.0

        # compute true positive rate = true positive rate / (true positive rate + false negative rate)
        true_pos_rate = true_pos[0] / (true_pos[0] + false_neg[0])

        # compute false positive rate = false positive rate / (false positive rate + true negative rate)
        false_pos_rate = false_pos[0] / (false_pos[0] + true_neg[0])

        # list to hold true and false positive rates separately
        rates = [true_pos_rate, false_pos_rate]

        # write roc curve data to file
        with open('output/text/{}_{}_roc.txt'.format(reg_grad.__name__, learning_rate), mode='a') as file:
            file.write('{},{}\n'.format(rates[0], rates[1]))

        # append matching true and false positive rates to true and false positive rates list
        true_false_rates.append((rates[0], rates[1]))

    # do some printing to show progress
    print('> Saved ROC data to file', '\n')

    # return true and false positive rates
    return true_false_rates


# computes auc
def comp_auc(true_false_rates):

    auc_sum = 0.0
    # for pairs x0, y0 and x1, y1 in false positive rates and true positive rates
    for (x0, y0), (x1, y1) in zip(true_false_rates[1:], true_false_rates):
        # sum (x0 + x1) * (y0 - y1)
        auc_sum += sum([(x0 + x1) * (y0 - y1)])

    # compute auc = 1.0 / 2 * auc_sum
    auc = 1.0 / 2 * auc_sum

    # do some printing to show progress
    print('> AUC:', auc, '\n')


########################################################################################################################
