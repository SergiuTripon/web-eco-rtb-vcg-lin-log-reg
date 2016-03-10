
########################################################################################################################


class Email(object):
    def __init__(self, features, label):
        # features
        self.features = features

        # label
        self.label = label


########################################################################################################################


def add_data(token, token_format):
    token = token.strip()
    if not isinstance(token, token_format):
        token = token_format(token)
        return token


########################################################################################################################


def load_file(file, file_format):
    print('\n> Loaded "{}"'.format(file), "\n")
    data = []
    with open(file, mode='r') as file:
            for line in file:
                tokens = line.split(",")
                added_data = [add_data(token, token_format) for token, token_format in zip(tokens, file_format)]
                data += [Email(added_data[:-1], added_data[-1])]

    return data


########################################################################################################################


# create empty list
data_format = []

# 48 continuous real [0,100] attributes of type word_freq_WORD
# 6 continuous real [0,100] attributes of type char_freq_CHAR
# 1 continuous real [1,...] attribute of type capital_run_length_average
# 1 continuous integer [1,...] attribute of type capital_run_length_longest
# 1 continuous integer [1,...] attribute of type capital_run_length_total
# 1 nominal {0,1} class attribute of type spam
data_format += 48 * [float] + 6 * [float] + 1 * [float] + 1 * [int] + 1 * [int] + 1 * [int]


########################################################################################################################
