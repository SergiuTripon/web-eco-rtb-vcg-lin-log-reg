
# python package
import operator

########################################################################################################################


# output object
class Output(object):

    def __init__(self, trained_weight, gold):

        # output weight
        self.weight = trained_weight

        # output gold
        self.gold = gold


########################################################################################################################

# computes roc curve data
def roc_curve(outputs, reg_grad, learning_rate):

    # sort outputs by weight attribute
    outputs = sorted(outputs, key=operator.attrgetter("weight"))

    # list to hold pairs of true and false positive rates
    true_false_rates = []

    # selects one output at a time
    # compares its weight with the weight of all the other outputs
    # for every current_output in outputs
    for current_output in outputs:

        # set rates start value to 0
        true_pos = [0.0]
        true_neg = [0.0]
        false_neg = [0.0]
        false_pos = [0.0]

        # for every output in outputs
        for output in outputs:
            # if output's gold is 1 and output's weight is larger or equal to current_output's weight
            if output.gold == 1 and output.weight >= current_output.weight:
                # increment true positive rate
                true_pos[0] += 1.0
            # if output's gold is 0 and output's weight is smaller than current_output's weight
            elif output.gold == 0 and output.weight < current_output.weight:
                # increment true negative rate
                true_neg[0] += 1.0
            # if output's gold is 0 and output's weight is larger or equal to current_output's weight
            elif output.gold == 0 and output.weight >= current_output.weight:
                # increment false positive rate
                false_pos[0] += 1.0
            # if output's gold is 1 and output's weight is smaller than current_output's weight
            elif output.gold == 1 and output.weight < current_output.weight:
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

        # append pairs of true and false positive rates to true_false_rates list
        true_false_rates.append((rates[0], rates[1]))

    # do some printing to show progress
    print('> Saved ROC data to file', '\n')

    # return true and false positive rates
    return true_false_rates


# computes auc
def comp_auc(true_false_rates):

    # reverse list holding the true false rates
    true_false_rates = true_false_rates[::-1]

    # set auc_sum start value to 0.0
    auc_sum = 0.0
    # true_false_rates[1:] - true_false_rates list missing the first pair, creating the k-1 effect
    # for pairs y0, x1 and y1, x0 in (true, false) - 1 and (true, false)
    for (y0, x1), (y1, x0) in zip(true_false_rates[1:], true_false_rates):
        # sum (xk - xk-1) * (yk + yk-1)
        auc_sum += sum([(x1 - x0) * (y1 + y0)])
    # compute auc = 1.0 / 2 * auc_sum
    auc = 1.0 / 2 * auc_sum
    # do some printing to show progress
    print('> AUC:', auc)


########################################################################################################################
