
# local python files
import eval
import proc

# python packages
import os

########################################################################################################################


# computes mean squared error
def mse(list1, list2):
    mean_squared_error = (sum([(x - y) ** 2 for x, y in zip(list1, list2)])) / float(len(list1))
    return mean_squared_error


########################################################################################################################


# trains the submitted regression with the submitted gradient descent
def train(train_set, reg_grad, learning_rate, threshold):

    # do some printing to show progress
    print('> Training', reg_grad.__name__, '\n')
    print('> Learning rate:', learning_rate, '\n')

    # set weights start value to 0
    weights = 57 * [0.0]

    # list to hold first weights
    first_weights = []
    # list to hold the golds
    golds = []
    # if the first 3 letters in the regression gradient function submitted are "lin"
    if reg_grad.__name__[:3] == "lin":
        # for every email in the train set
        for email in train_set:
            # compute first weights
            first_weights += [sum([x * y for x, y in zip(weights, email.attributes)])]
            # add every gold to golds list
            golds += [email.gold]
    # if the first 3 letters in the regression gradient function submitted are "lon"
    elif reg_grad.__name__[:3] == "log":
        # for every email in the train set
        for email in train_set:
            # compute first weights using logistic sigmoid
            first_weights += [proc.log_sigmoid(sum([x * y for x, y in zip(weights, email.attributes)]))]
            # add every gold to golds list
            golds += [email.gold]

    # compute the first mean squared error
    # used to compare with the next mean squared error
    # if next error is larger than the first error, gradient descent is diverging
    # if next error is smaller than the first error, gradient descent is converging
    first_mse = mse(first_weights, golds)

    # do some printing to show progress
    print('> Training started')

    # set epoch start value to 1
    epoch = 1

    # clear contents of output/text folder
    os.popen('rm -f ./output/text/*')

    # infinite while loop
    while epoch < 501:

        # compute next weights
        next_weights = reg_grad(weights, train_set, learning_rate)

        # list to hold updated weights
        updated_weights = []
        # list to hold the golds
        golds = []
        # if the first 3 letters in the regression gradient function submitted are "lin"
        if reg_grad.__name__[:3] == "lin":
            # for every email in the train set
            for email in train_set:
                # compute updated weights
                updated_weights += [sum([x * y for x, y in zip(next_weights, email.attributes)])]
                # add every gold to golds list
                golds += [email.gold]
        # if the first 3 letters in the regression gradient function submitted are "lon"
        elif reg_grad.__name__[:3] == "log":
            # for every email in the train set
            for email in train_set:
                # compute updated weights using logistic sigmoid
                updated_weights += [proc.log_sigmoid(sum([x * y for x, y in zip(next_weights, email.attributes)]))]
                # add every gold to golds list
                golds += [email.gold]

        # compute next_mse
        next_mse = mse(updated_weights, golds)

        # if next_mse is smaller or equal to first_mse and next_mse isn't smaller than the threshold
        # it means gradient descent is converging
        if next_mse <= first_mse and not (next_mse < threshold):

            # update first_mse as the next_mse
            # we are always checking if the next_mse is smaller than the previous one
            first_mse = next_mse
            # update weights as the next_weights
            weights = next_weights

            # do some printing to show progress
            print('> Epoch:', epoch, '| Train MSE is converging:', next_mse)

            # write current epoch and mse to file
            with open('output/text/{}_{}.txt'.format(reg_grad.__name__, learning_rate), mode='a') as file:
                file.write('{},{}\n'.format(epoch, next_mse))
            # increment epoch
            epoch += 1
        # else if next_mse is smaller or equal to first_mse and next_mse is smaller than the threshold
        elif next_mse <= first_mse and next_mse < threshold:
            # do some printing to show progress
            print('> Training finished\n'),
            print('> Saved training progress to file\n'),
            print('> Error vs. Threshold:', next_mse, '<', threshold)
            # break the while loop
            break
        # if next_mse isn't smaller or equal to first_mse
        # it means gradient descent is diverging
        else:
            # do some printing to show progress
            print('> Epoch:', epoch, '| Train MSE is diverging:', next_mse)
            # increment epoch
            epoch += 1

    # return weights
    return weights


########################################################################################################################


# tests the submitted regression with the submitted gradient descent
def test(trained_weights, test_set, reg_grad):

    # list to hold updated weights
    updated_weights = []
    # list to hold the golds
    golds = []
    # if the first 3 letters in the regression gradient function submitted are "lin"
    if reg_grad.__name__[:3] == "lin":
        # for every email in the train set
        for email in test_set:
            # compute updated weights
            updated_weights += [sum([x * y for x, y in zip(trained_weights, email.attributes)])]
            # add every gold to golds list
            golds += [email.gold]
    # if the first 3 letters in the regression gradient function submitted are "lon"
    elif reg_grad.__name__[:3] == "log":
        # for every email in the train set
        for email in test_set:
            # compute updated weights using logistic sigmoid
            updated_weights += [proc.log_sigmoid(sum([x * y for x, y in zip(trained_weights, email.attributes)]))]
            # add every gold to golds list
            golds += [email.gold]

    # compute test_mse
    test_mse = mse(updated_weights, golds)

    # do some printing to show progress
    print('\n> Testing', reg_grad.__name__)
    print('> Test MSE:', test_mse)
    print('> Testing finished', '\n')

    # list to hold results
    results = []
    # for every email in the test set
    for email in test_set:
        # compute updated weight
        updated_weight = sum([x * y for x, y in zip(trained_weights, email.attributes)])
        # add every email's updated_weight and gold to results list
        results += [eval.Result(updated_weight, email.gold)]
    # return results
    return results


########################################################################################################################
