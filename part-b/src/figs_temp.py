
# python packages
import os
import plotly.plotly as py
import plotly.graph_objs as go

########################################################################################################################


class IterFig(object):

    def __init__(self, epoch, error1):

        # epoch
        self.epoch = epoch

        # error1
        self.error1 = error1


########################################################################################################################


class ROCFig(object):
    def __init__(self, true_pos_rate1, false_pos_rate1):

        # true_pos_rate1
        self.true_pos_rate1 = true_pos_rate1

        # false_pos_rate1
        self.false_pos_rate1 = false_pos_rate1


########################################################################################################################


def load_result_file(result_file):
    data = []
    with open(result_file, mode='r') as file:
        for line in file:
            tokens = line.split(",")
            if "roc" in result_file:
                data += [ROCFig(tokens[0], tokens[1])]
            else:
                data += [IterFig(tokens[0], tokens[1])]

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

    # setup data
    data = [trace1]

    # setup layout
    layout = go.Layout(
        title=plot_title,
        xaxis=dict(
            title='Epoch',
            showlegend=True,
            showline=True,
            zeroline=False,
            range=[-500, 2000]
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

    # setup data
    data = [trace1]

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


def main():

    # iteration

    # lin_bgd
    # plot_title = "Linear Regression with Batch Gradient Descent"
    # result_file = "output/save/lin_bgd/lin_bgd_0.000001_2000.txt"
    # figure_file = "output/figs/lin_bgd/lin_bgd_0.000001_2000.png"

    # log_bgd
    # plot_title = "Logistic Regression with Batch Gradient Descent"
    # result_file = "output/save/log_bgd/log_bgd_0.0001_2000.txt"
    # figure_file = "output/figs/log_bgd/log_bgd_0.0001_2000.png"

    #################################################################################

    # roc
    # lin_bgd_roc
    # plot_title = "Linear Regression with Batch Gradient Descent - ROC Curve"
    # result_file = "output/save/lin_bgd/lin_bgd_0.000001_2000_roc.txt"
    # figure_file = "output/figs/lin_bgd/lin_bgd_0.000001_2000_roc.png"

    # log_bgd_roc
    plot_title = "Logistic Regression with Batch Gradient Descent - ROC Curve"
    result_file = "output/save/log_bgd/log_bgd_0.0001_2000_roc.txt"
    figure_file = "output/figs/log_bgd/log_bgd_0.0001_2000_roc.png"

    # plot_iter(plot_title, result_file, figure_file)
    plot_roc(plot_title, result_file, figure_file)


########################################################################################################################


if __name__ == '__main__':
    main()


########################################################################################################################
