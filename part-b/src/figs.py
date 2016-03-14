
# python packages
import os
import plotly.plotly as py
import plotly.graph_objs as go

########################################################################################################################


# iteration figure object
class IterFig(object):

    # constructor taking epoch, error1, error2, error3, error4 as arguments
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


# roc curve figure object
class ROCFig(object):

    # constructor taking true_pos_rate1, false_pos_rate1, true_pos_rate2, false_pos_rate2
    # true_pos_rate3, false_pos_rate4, true_pos_rate4, false_pos_rate4 as arguments
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


# loads output file
def load_file(output_file):
    # list to hold data
    data = []
    # open file
    with open(output_file, mode='r') as file:
        # for every line in file
        for line in file:
            # split line and assign results to tokens
            tokens = line.split(",")
            # if output_file contains "roc"
            if "roc" in output_file:
                # add each ROCFig object to the data list
                data += [ROCFig(tokens[0], tokens[1], tokens[2], tokens[3], tokens[4], tokens[5], tokens[6], tokens[7])]
            # if output_file doesn't contain "roc"
            else:
                # add each IterFig object to the data list
                data += [IterFig(tokens[0], tokens[1], tokens[2], tokens[3], tokens[4])]
    # return data
    return data


########################################################################################################################


# plots iteration figure
def plot_iter(plot_title, output_file, figure_file):

    # load data
    data = load_file(output_file)

    # list to hold epochs
    epochs = []
    # for every epoch in data
    for epoch in data:
        # append epoch to epochs list
        epochs.append(int(epoch.epoch.strip()))

    # list to hold error1
    error1 = []
    # for every epoch in data
    for error in data:
        # append error1 to error1 list
        error1.append(float(error.error1.strip()))

    # list to hold error2
    error2 = []
    # for every epoch in data
    for error in data:
        # append error2 to error2 list
        error2.append(float(error.error2.strip()))

    # list to hold error3
    error3 = []
    # for every epoch in data
    for error in data:
        # append error3 to error3 list
        error3.append(float(error.error3.strip()))

    # list to hold error4
    error4 = []
    # for every epoch in data
    for error in data:
        # append error4 to error4 list
        error4.append(float(error.error4.strip()))

    # setup trace 1
    trace1 = go.Scatter(
        x=epochs,
        y=error1,
        name="Learning rate: 0.01",
        mode="lines",
        line=dict(
            color='red'
        )
    )

    # setup trace 2
    trace2 = go.Scatter(
        x=epochs,
        y=error2,
        name="Learning rate: 0.001",
        mode="lines",
        line=dict(
            color='blue'
        )
    )

    # setup trace 3
    trace3 = go.Scatter(
        x=epochs,
        y=error3,
        name="Learning rate: 0.0001",
        mode="lines",
        line=dict(
            color='green'
        )
    )

    # setup trace 4
    trace4 = go.Scatter(
        x=epochs,
        y=error4,
        name="Learning rate: 0.00001",
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


# plots roc curve figure
def plot_roc(plot_title, output_file, figure_file):

    # load data
    data = load_file(output_file)

    # list to hold true_pos_rate1
    true_pos_rates1 = []
    # for every true_pos_rate in data
    for true_pos_rate in data:
        # append true_pos_rate1 to true_pos_rates1 list
        true_pos_rates1.append(float(true_pos_rate.true_pos_rate1.strip()))

    # list to hold false_pos_rate1
    false_pos_rates1 = []
    # for every false_pos_rate in data
    for false_pos_rate in data:
        # append false_pos_rate1 to false_pos_rates1 list
        false_pos_rates1.append(float(false_pos_rate.false_pos_rate1.strip()))

    # list to hold true_pos_rate2
    true_pos_rates2 = []
    # for every true_pos_rate in data
    for true_pos_rate in data:
        # append true_pos_rate2 to true_pos_rates2 list
        true_pos_rates2.append(float(true_pos_rate.true_pos_rate2.strip()))

    # list to hold false_pos_rate2
    false_pos_rates2 = []
    # for every false_pos_rate in data
    for false_pos_rate in data:
        # append false_pos_rate2 to false_pos_rates2 list
        false_pos_rates2.append(float(false_pos_rate.false_pos_rate2.strip()))

    # list to hold true_pos_rate3
    true_pos_rates3 = []
    # for every true_pos_rate in data
    for true_pos_rate in data:
        # append true_pos_rate3 to true_pos_rates3 list
        true_pos_rates3.append(float(true_pos_rate.true_pos_rate3.strip()))

    # list to hold false_pos_rate3
    false_pos_rates3 = []
    # for every false_pos_rate in data
    for false_pos_rate in data:
        # append false_pos_rate3 to false_pos_rates3 list
        false_pos_rates3.append(float(false_pos_rate.false_pos_rate3.strip()))

    # list to hold true_pos_rate4
    true_pos_rates4 = []
    # for every true_pos_rate in data
    for true_pos_rate in data:
        # append true_pos_rate4 to true_pos_rates4 list
        true_pos_rates4.append(float(true_pos_rate.true_pos_rate4.strip()))

    # list to hold false_pos_rate4
    false_pos_rates4 = []
    # for every false_pos_rate in data
    for false_pos_rate in data:
        # append false_pos_rate4 to false_pos_rates4 list
        false_pos_rates4.append(float(false_pos_rate.false_pos_rate1.strip()))

    # setup trace 1
    trace1 = go.Scatter(
        x=false_pos_rates1,
        y=true_pos_rates1,
        name="Learning rate: 0.01",
        mode="lines",
        line=dict(
            color='red'
        )
    )

    # setup trace 2
    trace2 = go.Scatter(
        x=false_pos_rates2,
        y=true_pos_rates2,
        name="Learning rate: 0.001",
        mode="lines",
        line=dict(
            color='blue'
        )
    )

    # setup trace 3
    trace3 = go.Scatter(
        x=false_pos_rates3,
        y=true_pos_rates3,
        name="Learning rate: 0.0001",
        mode="lines",
        line=dict(
            color='green'
        )
    )

    # setup trace 4
    trace4 = go.Scatter(
        x=false_pos_rates4,
        y=true_pos_rates4,
        name="Learning rate: 0.00001",
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
            x=0.68,
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


# main function
def main():

    # iteration
    # lin_sgd
    plot_title = "Linear Regression with Stochastic Gradient Descent"
    output_file = "output/save/lin_sgd/lin_sgd_all.txt"
    figure_file = "output/figs/lin_sgd/lin_sgd_all.png"

    # lin_bgd
    # plot_title = "Linear Regression with Batch Gradient Descent"
    # output_file = "output/save/lin_bgd/lin_bgd_all.txt"
    # figure_file = "output/figs/lin_bgd/lin_bgd_all.png"

    # log_sgd
    # plot_title = "Logistic Regression with Stochastic Gradient Descent"
    # output_file = "output/save/log_sgd/log_sgd_all.txt"
    # figure_file = "output/figs/log_sgd/log_sgd_all.png"

    # log_bgd
    # plot_title = "Logistic Regression with Batch Gradient Descent"
    # output_file = "output/save/log_bgd/log_bgd_all.txt"
    # figure_file = "output/figs/log_bgd/log_bgd_all.png"

    #################################################################################

    # roc
    # lin_sgd_roc
    # plot_title = "Linear Regression with Stochastic Gradient Descent - ROC Curve"
    # output_file = "output/save/lin_sgd/lin_sgd_all_roc.txt"
    # figure_file = "output/figs/lin_sgd/lin_sgd_all_roc.png"

    # lin_bgd_roc
    # plot_title = "Linear Regression with Batch Gradient Descent - ROC Curve"
    # output_file = "output/save/lin_bgd/lin_bgd_all_roc.txt"
    # figure_file = "output/figs/lin_bgd/lin_bgd_all_roc.png"

    # log_sgd_roc
    # plot_title = "Logistic Regression with Stochastic Gradient Descent - ROC Curve"
    # output_file = "output/save/log_sgd/log_sgd_all_roc.txt"
    # figure_file = "output/figs/log_sgd/log_sgd_all_roc.png"

    # log_bgd_roc
    # plot_title = "Logistic Regression with Batch Gradient Descent - ROC Curve"
    # output_file = "output/save/log_bgd/log_bgd_all_roc.txt"
    # figure_file = "output/figs/log_bgd/log_bgd_all_roc.png"

    plot_iter(plot_title, output_file, figure_file)
    # plot_roc(plot_title, output_file, figure_file)


########################################################################################################################


# runs main function
if __name__ == '__main__':
    main()


########################################################################################################################
