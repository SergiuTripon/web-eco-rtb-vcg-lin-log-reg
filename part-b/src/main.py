
# local python files
import run
import proc


########################################################################################################################


def main():

    # learning rates

    # lin_sgd
    learning_rate = 0.00001
    # learning_rate = 0.0001
    # learning_rate = 0.001
    # learning_rate = 0.01

    # lin_bgd
    # learning_rate = 0.000001

    # log_sgd
    # learning_rate = 0.001

    # log_bgd
    # learning_rate = 0.0001

    # the four learners
    # lin_sgd_threshold = 0.1
    # lin_bgd_threshold = 0.1
    # log_sgd_threshold = 0.1
    # log_bgd_threshold = 0.1

    # run linear regression with stochastic gradient descent
    run.run(proc.lin_sgd, learning_rate, lin_sgd_threshold)

    # run linear regression with batch gradient descent
    # run.run(proc.lin_bgd, learning_rate, lin_bgd_threshold)

    # run logistic regression with stochastic gradient descent
    # run.run(proc.log_sgd, learning_rate, log_sgd_threshold)

    # run logistic regression with batch gradient descent
    # run.run(proc.log_bgd, learning_rate, log_bgd_threshold)


########################################################################################################################

# runs main class
if __name__ == '__main__':
    main()


########################################################################################################################
