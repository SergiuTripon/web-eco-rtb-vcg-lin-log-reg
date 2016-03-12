
# python packages
import os
import plotly.offline as py
import plotly.graph_objs as go
from matplotlib import pyplot
from matplotlib import patches

########################################################################################################################


class IterFig(object):

    def __init__(self, epoch, error):

        # result weight
        self.epoch = epoch

        # result gold
        self.error = error


class ROCFig(object):
    def __init__(self, true_pos_rate, false_pos_rate):

        # result weight
        self.true_pos_rate = true_pos_rate

        # result gold
        self.false_pos_rate = false_pos_rate


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


def plot_iter(result_file, plot_title):

    # load data
    data = load_result_file("output/text/" + result_file + ".txt")

    epochs = []
    for epoch in data:
        epochs.append(int(epoch.epoch.strip()))

    errors = []
    for error in data:
        errors.append(float(error.error.strip()))

    # sets the figure_file naming to be same as result_file
    figure_file = "output/figs/{}".format(result_file)

    # create a figure
    fig = pyplot.figure()

    # plot data
    pyplot.scatter(epochs, errors, color="#333333")

    # add grid
    pyplot.grid(True)

    # add labels and color
    pyplot.title(plot_title, color="#333333")
    pyplot.ylabel("Error", color="#333333")
    pyplot.xlabel("Epoch", color="#333333")

    # add legend
    black_patch = patches.Patch(color="#333333", label="Learning rate: 0.001")
    pyplot.legend(handles=[black_patch], fancybox=True)

    # clear contents of output/figs folder
    os.popen("rm -f ./output/figs/*")

    # save plot as a png image
    fig.savefig(figure_file + ".png")


########################################################################################################################


def plot_roc(result_file, plot_title):

    # load data
    data = load_result_file("output/text/" + result_file + ".txt")

    true_pos_rates = []
    for true_pos_rate in data:
        true_pos_rates.append(float(true_pos_rate.true_pos_rate.strip()))

    false_pos_rates = []
    for false_pos_rate in data:
        false_pos_rates.append(float(false_pos_rate.false_pos_rate.strip()))

    # sets the figure_file naming to be same as result_file
    figure_file = "output/figs/{}".format(result_file)

    # create a figure
    fig = pyplot.figure()

    # plot data
    pyplot.plot(false_pos_rates, true_pos_rates, color="#333333")

    # add grid
    pyplot.grid(True)

    # add labels and color
    pyplot.title(plot_title, color="#333333")
    pyplot.ylabel("True Positive Rate", color="#333333")
    pyplot.xlabel("False Positive Rate", color="#333333")

    # add legend
    black_patch = patches.Patch(color="#333333", label="Learning rate: 0.001")
    pyplot.legend(handles=[black_patch], loc=4)

    # clear contents of output/figs folder
    os.popen("rm -f ./output/figs/*")

    # save plot as a png image
    fig.savefig(figure_file + ".png")


########################################################################################################################


def main():

    # iteration
    # lin_sgd
    plot_title = "Linear Regression with SGD"
    result_file = "lin_sgd_0.001"

    # lin_bgd
    # plot_title = "Linear Regression with BGD"
    # result_filename = "lin_bgd_0.001"

    # log_sgd
    # plot_title = "Logistic Regression with SGD"
    # result_filename = "log_sgd_0.001"

    # log_bgd
    # plot_title = "Logistic Regression with BGD"
    # result_filename = "log_bgd_0.001"

    # roc
    # lin_sgd_roc
    # plot_title = "Linear Regression with SGD - ROC Curve"
    # result_file = "lin_sgd_0.0001_roc"

    # lin_bgd_roc
    # plot_title = "Linear Regression with BGD - ROC Curve"
    # result_file = "lin_bgd_0.001_roc"

    # log_sgd_roc
    # plot_title = "Logistic Regression with SGD - ROC Curve"
    # result_file = "log_sgd_0.001_roc"

    # log_bgd_roc
    # plot_title = "Logistic Regression with BGD - ROC Curve"
    # result_file = "log_bgd_0.001_roc"

    plot_iter(result_file, plot_title)
    # plot_roc(result_file, plot_title)

    '''

    # Create a trace
    trace = go.Scatter(
        x=data[0],
        y=data[1],
        name="Learning rate: 0.0001",
        mode="markers",
        marker=dict(
            size='10',
            color='#333333',
        )
    )

    data = [trace]

    layout = go.Layout(
        title=plot_title,
        font=dict(
            family='Times New Roman, Times, serif',
            size=18,
            color='#333333'
        ),
        showlegend=True,
        legend=dict(
            x=0.9,
            y=1,
            traceorder='normal',
            font=dict(
                family='Times New Roman, Times, serif',
                size=12,
                color='#333333'
            ),
            bgcolor='#E3E3E3',
            bordercolor='#333333',
            borderwidth=1
        ),
        paper_bgcolor='#F9F9F9',
        plot_bgcolor='#F9F9F9',
        xaxis=dict(
            title='Epoch',
            titlefont=dict(
                family='Times New Roman, Times, serif',
                size=18,
                color='#333333'
            ),
            showline=True,
            showticklabels=True,
            tickfont=dict(
                family='Times New Roman, Times, serif',
                size=14,
                color='#333333'
            ),
        ),
        yaxis=dict(
            title='Error',
            titlefont=dict(
                family='Times New Roman, Times, serif',
                size=18,
                color='#333333'
            ),
            showticklabels=True,
            tickfont=dict(
                family='Times New Roman, Times, serif',
                size=14,
                color='#333333'
            ),
        ),
    )

    fig = go.Figure(data=data, layout=layout)

    py.plot(fig, filename=figure_filename + ".html")

    '''

########################################################################################################################


if __name__ == '__main__':
    main()


########################################################################################################################
