"""
This file is used to analyse the dialog acts data
"""
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from a.preprocessing import *


def plot_label_distribution(data):
    """
    This function plots the label distributions of train and test data in a histogram
    and saves this image as a png
    :param data: [[label1, utt1], [label2, utt2], ...]
    :return: /
    """
    f, ax = plt.subplots(figsize=(10, 5))

    dictionary = Counter([data[i][0] for i in range(len(data))])
    plt.barh(list(dictionary.keys()), dictionary.values(), label='test data')

    train, test = train_test_split(data, test_size=0.15, random_state=42)
    train_dict = Counter([train[i][0] for i in range(len(train))])
    plt.barh(list(train_dict.keys()), train_dict.values(), label='train data')

    test_dict = Counter([test[i][0] for i in range(len(test))])
    # plt.bar(list(test_dict.keys()), test_dict.values(), color='r')

    for i, v in enumerate(dictionary.values()):
        ax.text(v + 100, i - .25, str(v), fontweight='bold')

    plt.title('label distribution')
    plt.legend(loc='best')
    plt.ylabel('data labels')
    plt.xlabel('number of accurancies')
    plt.savefig('label_distribution.png')

    plt.show()


def plot_utterance_length(data):
    """
    This function plots the average utterance length per label and it's standard deviation
    :param data: [[label1, utt1], [label2, utt2], ...]
    :return: /
    """
    # put utterances with same label together in dict
    data_dict = defaultdict(list)

    # populate dict with utterance lengths
    for label, utt in data:
        data_dict[label].append(len(utt.split(' ')))

    means = [np.mean(np.array(utt_len)) for utt_len in data_dict.values()]
    standard_dev = [np.std(np.array(utt_len)) for utt_len in data_dict.values()]

    plt.barh(list(data_dict.keys()), means, xerr=standard_dev, label='test data', alpha=0.8, align='center', capsize=5)
    plt.title('Average number of words in utterances')
    plt.ylabel('data labels')
    plt.xlabel('utterance lengths')
    plt.savefig('utterance_length.png')
    plt.show()


def out_of_voc_words(data):
    """
    Makes a dict of out of voc words per label
    :param data: [[label1, utt1], [label2, utt2], ...]
    :return: label to set of out of voc words dict
    """
    train, test = train_test_split(data, test_size=0.15, random_state=42)

    x_train, y_train = split_x_y(train)

    # default regex pattern only matches groups with more than 2 characters
    features = CountVectorizer()
    features.fit(x_train)

    feature_words = features.get_feature_names()

    d = defaultdict(set)
    for label, utt in test:
        d[label].update([word for word in utt.split(' ') if word not in feature_words and len(word) > 1])

    return d


if __name__ == '__main__':
    data = read_data("dialog_acts.dat")

    print(out_of_voc_words(data))
