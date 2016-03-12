
# python packages
import os
import plotly.offline as py
import plotly.graph_objs as go
from matplotlib import pyplot
from matplotlib import patches

########################################################################################################################


class FileResult(object):

    def __init__(self, epoch, error):

        # result weight
        self.epoch = epoch

        # result gold
        self.error = error


def load_result_file(file):
    data = []
    with open(file, mode='r') as file:
        for line in file:
            tokens = line.split(",")
            data += [FileResult(tokens[0], tokens[1])]

    return data


########################################################################################################################


def main():

    # lin_sgd
    plot_title = "Linear Regression with Stochastic Gradient Descent"
    result_filename = "lin_sgd_0.001"

    # lin_bgd
    # plot_title = "Linear Regression with Batch Gradient Descent"
    # result_filename = "lin_bgd_0.001"

    # log_sgd
    # plot_title = "Logistic Regression with Stochastic Gradient Descent"
    # result_filename = "log_sgd_0.001"

    # log_bgd
    # plot_title = "Logistic Regression with Batch Gradient Descent"
    # result_filename = "log_bgd_0.001"

    # load data
    data = load_result_file("output/text/" + result_filename + ".txt")

    # sets the figure file name to be same as result file name
    figure_filename = "output/figs/{}".format(result_filename)

    fig = pyplot.figure()

    epochs = []
    for epoch in data:
        epochs.append(int(epoch.epoch))

    errors = []
    for error in data:
        errors.append(float(error.error))

    # plots data
    pyplot.scatter(epochs, errors, color="#333333")

    # adds grid
    pyplot.grid(True)

    # labels
    pyplot.title(plot_title, color="#333333")
    pyplot.ylabel("Error", color="#333333")
    pyplot.xlabel("Epoch", color="#333333")

    # legend
    black_patch = patches.Patch(color="#333333", label="Learning rate: 0.001")
    pyplot.legend(handles=[black_patch], fancybox=True)

    # clears contents of figs folder
    os.popen("rm -f ./output/figs/*")

    # saves plot as png
    fig.savefig(figure_filename + ".png")

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
