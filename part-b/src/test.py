
# local python files
import eval
import proc

# python packages
import os

########################################################################################################################


# compute mean squared error
def mse(list1, list2):
    return (sum([(x - y) ** 2 for x, y in zip(list1, list2)])) / float(len(list1))


########################################################################################################################


def train(weights, train_set, reg_model, reg_gradient, grad_func, learning_rate, threshold):

    print('> Learning rate:', learning_rate)

    # initialize the error
    res1 = []
    res2 = []
    if grad_func.__name__[:3] == "lin":
        for email in train_set:
            res1 += [sum([x * y for x, y in zip(weights, email.features)])]
            res2 += [email.label]
    elif grad_func.__name__[:3] == "log":
        for email in train_set:
            res1 += [proc.log_sigmoid(sum([x * y for x, y in zip(weights, email.features)]))]
            res2 += [email.label]

    start_error = mse(res1, res2)

    print('> 1st Training MSE:', start_error, '\n')
    print('> Training started')

    epoch = 1

    # loop
    epoch1 = []
    new_error1 = []
    os.popen('rm -f ./output/text/*')
    while True:

        # calculate new weights
        new_weights = grad_func(weights, train_set, reg_model, reg_gradient, learning_rate)

        # calculate new error
        res1 = []
        res2 = []
        if grad_func.__name__[:3] == "lin":
            for email in train_set:
                res1 += [sum([x * y for x, y in zip(new_weights, email.features)])]
                res2 += [email.label]
        elif grad_func.__name__[:3] == "log":
            for email in train_set:
                res1 += [proc.log_sigmoid(sum([x * y for x, y in zip(new_weights, email.features)]))]
                res2 += [email.label]

        new_error = mse(res1, res2)

        if new_error <= start_error:
            print('> Epoch:', epoch, '| Train MSE is converging:', new_error)
            with open('output/text/{}_{}.txt'.format(grad_func.__name__, learning_rate), mode='a') as fd:
                fd.write('{},{}\n'.format(epoch, new_error))
            epoch1.append(epoch)
            new_error1.append(new_error)
            epoch += 1
        else:
            print('> Epoch:', epoch, '| Train MSE is diverging:', new_error)
            epoch += 1

        if new_error <= start_error:
            start_error = new_error
            weights = new_weights

            if new_error < threshold:
                print('> Training finished\n'),
                print('> Error vs. Threshold:', new_error, '<', threshold)
                break

    # done
    return weights


########################################################################################################################


def test(test_set, train_set, reg_model, reg_grad, grad_func, learning_rate, threshold):

    # train
    init_weights = 57 * [0.0]
    weights = train(init_weights, train_set, reg_model, reg_grad, grad_func, learning_rate, threshold)

    # test
    res1 = []
    res2 = []
    if grad_func.__name__[:3] == "lin":
        for email in test_set:
            res1 += [sum([x * y for x, y in zip(weights, email.features)])]
            res2 += [email.label]
    elif grad_func.__name__[:3] == "log":
        for email in test_set:
            res1 += [proc.log_sigmoid(sum([x * y for x, y in zip(weights, email.features)]))]
            res2 += [email.label]

    test_error = mse(res1, res2)

    print('\n> Test started')
    print('> Testing MSE:', test_error)
    print('> Test finished')
    print('\n##########################################################################')

    # produce a result set
    results = [eval.DataResult(data_point.label, reg_model(weights, data_point.features))
               for data_point in test_set]

    # output roc data
    roc = eval.rocdata(results)
    auc = eval.auc(roc)
    with open('output/text/{}_{}_auc={}.txt'.format(grad_func.__name__, learning_rate, auc).lower(),
              mode='w') as fd:
        for fpr, tpr in roc:
            fd.write('{}, {}\n'.format(fpr, tpr))


########################################################################################################################
