
########################################################################################################################


# email object
class Email(object):
    def __init__(self, attributes, gold):

        # email attributes
        self.attributes = attributes

        # email gold
        self.gold = gold


########################################################################################################################


# adds data to list after ensuring correct data format
def add_data(token, token_format):
    # strip token
    token = token.strip()
    # if token isn't an instance of its matching token_format
    if not isinstance(token, token_format):
        # change token's type to matching token_format
        token = token_format(token)
        # return token
        return token


########################################################################################################################


# loads input file
def load_file(file, file_format):
    # do some printing to show progress
    print('\n> Loaded "{}"'.format(file), "\n")
    # list to hold data
    data = []
    # open file
    with open(file, mode='r') as file:
            # for every line in file
            for line in file:
                # split line and assign results to tokens
                tokens = line.split(",")
                # for every token and token format, call the add_data function, and assign results to added_data
                added_data = [add_data(token, token_format) for token, token_format in zip(tokens, file_format)]
                # add added_data's elements apart from the last 1 as the first attribute of the Email object
                # add added_data's last element as the second attribute of the Email object
                # add each Email object to the data list
                data += [Email(added_data[:-1], added_data[-1])]
    # return data
    return data


########################################################################################################################


# list to hold data format
data_format = []

# 48 continuous real [0,100] attributes of type word_freq_WORD
# 6 continuous real [0,100] attributes of type char_freq_CHAR
# 1 continuous real [1,...] attribute of type capital_run_length_average
# 1 continuous integer [1,...] attribute of type capital_run_length_longest
# 1 continuous integer [1,...] attribute of type capital_run_length_total
# 1 nominal {0,1} class attribute of type spam

# looking at the data, I discovered that the first 48 elements of each line were not of a uniform data type
# e.g. some were int and some were float which can cause problems in various ways
# therefore, I decided to convert them to float
# the same discovery occurred for the following 6 and 1 elements after the initial 48
# only the last 3 elements had a uniform data type which was int
data_format += 48 * [float] + 6 * [float] + 1 * [float] + 1 * [int] + 1 * [int] + 1 * [int]


########################################################################################################################
