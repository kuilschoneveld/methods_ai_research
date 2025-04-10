from nltk.corpus import stopwords
from sklearn import preprocessing


def read_data(data_path):
    """
    This function opens a data file
    and returns the first word of each sentence as label
    and the rest of the sentence as data
    :param data_path:  the path of the datafile
    :return: [[label1, data1], [label2, data2], ...]
    """
    data = []
    # read in data
    with open(data_path) as dataset:
        for line in dataset.readlines():
            data.append(line.strip("\n").split(' ', 1))

    return data


def split_x_y(data):
    """
    splits the data and labels of a double data list
    :param data: of the form [[label1, x_data1], [label2, x_data2], ...]
    :return: [x_data1, x_data2, ...] and [label1, label2, ...]
    """
    # split x and y in different lists (easier to use for build in methods)
    y_data = [data_point[0] for data_point in data]
    x_data = [data_point[1] for data_point in data]

    return x_data, y_data


def remove_stop_words(sentences):
    """
    remove the stopwords in the sentences
    :param sentences: list of sentences
    :return: list of sentences without stopwords
    """
    sentences_new = []

    is_stopword = stopwords.words('english').__contains__
    for sentence in sentences:
        sentences_new.append(" ".join([word for word in sentence.split(' ')
                                       if not is_stopword(word)]))
    return sentences_new


def to_one_hot(y_data):
    lb = preprocessing.LabelBinarizer()
    one_hot_labels = lb.fit_transform(y_data)

    return one_hot_labels
