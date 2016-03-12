
# local python files
import run
import proc


########################################################################################################################


def main():

    # learning rates

    # lin_sgd
    # learning_rate = 0.00001
    # learning_rate = 0.0001
    learning_rate = 0.001
    # learning_rate = 0.01

    # lin_bgd
    # learning_rate = 0.0000001
    # learning_rate = 0.000001
    # learning_rate = 0.00001
    # learning_rate = 0.0001
    # learning_rate = 0.001
    # learning_rate = 0.01

    # log_sgd
    # learning_rate = 0.0001
    # learning_rate = 0.001
    # learning_rate = 0.01
    # learning_rate = 0.1

    # log_bgd
    # learning_rate = 0.0000001
    # learning_rate = 0.000001
    # learning_rate = 0.00001
    # learning_rate = 0.0001
    # learning_rate = 0.001
    # learning_rate = 0.01
    # learning_rate = 0.1

    # threshold
    threshold = 0.1

    # run linear regression with stochastic gradient descent
    run.run(proc.lin_sgd, learning_rate, threshold)

    # run linear regression with batch gradient descent
    # run.run(proc.lin_bgd, learning_rate, threshold)

    # run logistic regression with stochastic gradient descent
    # run.run(proc.log_sgd, learning_rate, threshold)

    # run logistic regression with batch gradient descent
    # run.run(proc.log_bgd, learning_rate, threshold)


########################################################################################################################

# runs main class
if __name__ == '__main__':
    main()


########################################################################################################################
