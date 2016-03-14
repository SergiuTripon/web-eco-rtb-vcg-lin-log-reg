
# python packages
import os
import plotly.plotly as py
import plotly.graph_objs as go

########################################################################################################################


class IterFig(object):

    def __init__(self, epoch, error1, error2, error3, error4):

        # epoch
        self.epoch = epoch

        # error1
        self.error1 = error1
        # error2
        self.error2 = error2
        # error3
        self.error3 = error3
        # error4
        self.error4 = error4


########################################################################################################################


class ROCFig(object):
    def __init__(self, true_pos_rate1, false_pos_rate1, true_pos_rate2, false_pos_rate2,
                 true_pos_rate3, false_pos_rate3, true_pos_rate4, false_pos_rate4):

        # true_pos_rate1
        self.true_pos_rate1 = true_pos_rate1

        # false_pos_rate1
        self.false_pos_rate1 = false_pos_rate1

        # true_pos_rate2
        self.true_pos_rate2 = true_pos_rate2

        # false_pos_rate2
        self.false_pos_rate2 = false_pos_rate2

        # true_pos_rate3
        self.true_pos_rate3 = true_pos_rate3

        # false_pos_rate3
        self.false_pos_rate3 = false_pos_rate3

        # true_pos_rate4
        self.true_pos_rate4 = true_pos_rate4

        # false_pos_rate4
        self.false_pos_rate4 = false_pos_rate4


########################################################################################################################


def load_result_file(result_file):
    data = []
    with open(result_file, mode='r') as file:
        for line in file:
            tokens = line.split(",")
            if "roc" in result_file:
                data += [ROCFig(tokens[0], tokens[1], tokens[2], tokens[3], tokens[4], tokens[5], tokens[6], tokens[7])]
            else:
                data += [IterFig(tokens[0], tokens[1], tokens[2], tokens[3], tokens[4])]

    return data


########################################################################################################################


def plot_iter(plot_title, result_file, figure_file):

    # load data
    data = load_result_file(result_file)

    epochs = []
    for epoch in data:
        epochs.append(int(epoch.epoch.strip()))

    error1 = []
    for error in data:
        error1.append(float(error.error1.strip()))

    error2 = []
    for error in data:
        error2.append(float(error.error2.strip()))

    error3 = []
    for error in data:
        error3.append(float(error.error3.strip()))

    error4 = []
    for error in data:
        error4.append(float(error.error4.strip()))

    # setup trace 1
    trace1 = go.Scatter(
        x=epochs,
        y=error1,
        name="Learning rate: 0.0001",
        mode="lines",
        line=dict(
            color='red'
        )
    )

    # setup trace 2
    trace2 = go.Scatter(
        x=epochs,
        y=error2,
        name="Learning rate: 0.00001",
        mode="lines",
        line=dict(
            color='blue'
        )
    )

    # setup trace 3
    trace3 = go.Scatter(
        x=epochs,
        y=error3,
        name="Learning rate: 0.000001",
        mode="lines",
        line=dict(
            color='green'
        )
    )

    # setup trace 4
    trace4 = go.Scatter(
        x=epochs,
        y=error4,
        name="Learning rate: 0.0000001",
        mode="lines",
        line=dict(
            color='orange'
        )
    )

    # setup data
    data = [trace1, trace2, trace3, trace4]

    # setup layout
    layout = go.Layout(
        title=plot_title,
        xaxis=dict(
            title='Epoch',
            showline=True,
            zeroline=False,
            range=[-10, 50]
        ),
        yaxis=dict(
            title='Mean Squared Error',
            showline=True
        ),
        legend=dict(
            x=0.68,
            y=1,
            borderwidth=1
        )
    )

    # setup figure
    fig = go.Figure(data=data, layout=layout)

    # clear contents of output/figs folder
    # os.popen("rm -f ./output/figs/*")

    # save figure
    py.image.save_as(fig, filename=figure_file)

########################################################################################################################


def plot_roc(plot_title, result_file, figure_file):

    # load data
    data = load_result_file(result_file)

    #
    true_pos_rates1 = []
    for true_pos_rate in data:
        true_pos_rates1.append(float(true_pos_rate.true_pos_rate1.strip()))

    false_pos_rates1 = []
    for false_pos_rate in data:
        false_pos_rates1.append(float(false_pos_rate.false_pos_rate1.strip()))

    true_pos_rates2 = []
    for true_pos_rate in data:
        true_pos_rates2.append(float(true_pos_rate.true_pos_rate2.strip()))

    false_pos_rates2 = []
    for false_pos_rate in data:
        false_pos_rates2.append(float(false_pos_rate.false_pos_rate2.strip()))

    true_pos_rates3 = []
    for true_pos_rate in data:
        true_pos_rates3.append(float(true_pos_rate.true_pos_rate3.strip()))

    false_pos_rates3 = []
    for false_pos_rate in data:
        false_pos_rates3.append(float(false_pos_rate.false_pos_rate3.strip()))

    true_pos_rates4 = []
    for true_pos_rate in data:
        true_pos_rates4.append(float(true_pos_rate.true_pos_rate4.strip()))

    false_pos_rates4 = []
    for false_pos_rate in data:
        false_pos_rates4.append(float(false_pos_rate.false_pos_rate1.strip()))

    # setup trace 1
    trace1 = go.Scatter(
        x=false_pos_rates1,
        y=true_pos_rates1,
        name="Learning rate: 0.0001",
        mode="lines",
        line=dict(
            color='red'
        )
    )

    # setup trace 2
    trace2 = go.Scatter(
        x=false_pos_rates2,
        y=true_pos_rates2,
        name="Learning rate: 0.00001",
        mode="lines",
        line=dict(
            color='blue'
        )
    )

    # setup trace 3
    trace3 = go.Scatter(
        x=false_pos_rates3,
        y=true_pos_rates3,
        name="Learning rate: 0.000001",
        mode="lines",
        line=dict(
            color='green'
        )
    )

    # setup trace 4
    trace4 = go.Scatter(
        x=false_pos_rates4,
        y=true_pos_rates4,
        name="Learning rate: 0.0000001",
        mode="lines",
        line=dict(
            color='orange'
        )
    )

    # setup data
    data = [trace1, trace2, trace3, trace4]

    # setup layout
    layout = go.Layout(
        title=plot_title,
        showlegend=True,
        xaxis=dict(
            title='False Positive Rate',
            showline=True,
            zeroline=False
        ),
        yaxis=dict(
            title='True Positive Rate',
            showline=True,
            zeroline=False
        ),
        legend=dict(
            x=0.65,
            y=0.1,
            borderwidth=1
        )
    )

    # setup figure
    fig = go.Figure(data=data, layout=layout)

    # clear contents of output/figs folder
    # os.popen("rm -f ./output/figs/*")

    # save figure
    py.image.save_as(fig, filename=figure_file)

########################################################################################################################


def main():

    # iteration
    # lin_sgd
    # plot_title = "Linear Regression with Stochastic Gradient Descent"
    # result_file = "output/save/lin_sgd/lin_sgd_all.txt"
    # figure_file = "output/figs/lin_sgd/lin_sgd_all.png"

    # lin_bgd
    # plot_title = "Linear Regression with Batch Gradient Descent"
    # result_file = "output/save/lin_bgd/lin_bgd_all.txt"
    # figure_file = "output/figs/lin_bgd/lin_bgd_all.png"

    # log_sgd
    # plot_title = "Logistic Regression with Stochastic Gradient Descent"
    # result_file = "output/save/log_sgd/log_sgd_all.txt"
    # figure_file = "output/figs/log_sgd/log_sgd_all.png"

    # log_bgd
    # plot_title = "Logistic Regression with Batch Gradient Descent"
    # result_file = "output/save/log_bgd/log_bgd_all.txt"
    # figure_file = "output/figs/log_bgd/log_bgd_all.png"

    #################################################################################

    # roc
    # lin_sgd_roc
    # plot_title = "Linear Regression with Stochastic Gradient Descent - ROC Curve"
    # result_file = "output/save/lin_sgd/lin_sgd_all_roc.txt"
    # figure_file = "output/figs/lin_sgd/lin_sgd_all_roc.png"

    # lin_bgd_roc
    # plot_title = "Linear Regression with Batch Gradient Descent - ROC Curve"
    # result_file = "output/save/lin_bgd/lin_bgd_all_roc.txt"
    # figure_file = "output/figs/lin_bgd/lin_bgd_all_roc.png"

    # log_sgd_roc
    # plot_title = "Logistic Regression with Stochastic Gradient Descent - ROC Curve"
    # result_file = "output/save/log_sgd/log_sgd_all_roc.txt"
    # figure_file = "output/figs/log_sgd/log_sgd_all_roc.png"

    # log_bgd_roc
    plot_title = "Logistic Regression with Batch Gradient Descent - ROC Curve"
    result_file = "output/save/log_bgd/log_bgd_all_roc.txt"
    figure_file = "output/figs/log_bgd/log_bgd_all_roc.png"

    # plot_iter(plot_title, result_file, figure_file)
    plot_roc(plot_title, result_file, figure_file)


########################################################################################################################


if __name__ == '__main__':
    main()


########################################################################################################################
