
# python package
import operator

########################################################################################################################


class Result(object):

    def __init__(self, label, trained_weight, prediction):

        self.label = label

        self.weight = trained_weight

        self.prediction = prediction


########################################################################################################################

# computes roc curve data
def roc_curve(results, reg_grad, learning_rate):

    # sort results by weight
    results = sorted(results, key=operator.attrgetter("weight"))

    # variable to hold true false rates
    true_false_rates = []

    # make the rest such that on the final one, for all dr, prediction is -
    for result in results:

        # initialize
        true_pos = [0.0]
        true_neg = [0.0]
        false_neg = [0.0]
        false_pos = [0.0]

        for result1 in results:
            if result1.label == 1 and result1.weight >= result.weight:
                true_pos[0] += 1.0
            elif result1.label == 0 and result1.weight < result.weight:
                true_neg[0] += 1.0
            elif result1.label == 0 and result1.weight >= result.weight:
                false_pos[0] += 1.0
            elif result1.label == 1 and result1.weight < result.weight:
                false_neg[0] += 1.0

        # compute true positive rate
        true_pos_rate = true_pos[0] / (false_neg[0] + true_pos[0])

        # compute false positive rate
        false_pos_rate = false_pos[0] / (false_pos[0] + true_neg[0])

        # compile all rates in one list
        rates = [true_pos_rate, false_pos_rate]

        # write roc data to file
        with open('output/text/{}_{}_roc.txt'.format(reg_grad.__name__, learning_rate), mode='a') as file:
            file.write('{},{}\n'.format(rates[0], [1]))

        true_false_rates.append((rates[0], rates[1]))

    print('> Saved ROC data to file', '\n')

    return true_false_rates


# computes auc
def comp_auc(true_false_rates):

    res1 = 0.0
    for (x0, y0), (x1, y1) in zip(true_false_rates[1:], true_false_rates):
        res1 += sum([(x0 + x1) * (y0 - y1)])

    auc = 1.0 / 2 * res1
    return auc


########################################################################################################################
