from collections import Counter
from sklearn.model_selection import train_test_split
import sys
import copy
# nltk.download('stopwords')
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

try:
    from a.preprocessing import *            
except ImportError:
    try:
        from preprocessing import *                   
    except ImportError as error:
        print("ERROR: There was an error loading required modules. Please make sure that all modules are in the correct folders.")
        print(error)


categories = []
most_common_category = ''
category_lengths = {}
category_ratios = {}
train = []
test = []

def init():
    global categories, most_common_category, category_lengths, category_ratios, train, test


def most_common(data):
    """
    Determines the most common category in the database and returns it.

    :param data: the database
    :return: the most common category of the database

    """
    global categories
    number = Counter([data[i][0] for i in range(len(data))])
    categories = list(number.keys())
    return max(number, key=number.get)


def category_ranking(data):
    """
    Creates a list of lists that shows the number of times each word appeared per category and returns it.

    :param data: the database
    :return: the list of lists containing words and the number of times they appear per category

    """
    global category_lengths

    cat_list = {}
    cat_list["all"] = Counter([word for line in data for word in line[1].split(" ")])

    for cat in categories:
        category_lengths[cat] = len([data[i][0] for i in range(len(data)) if data[i][0] == cat])
        cat_list[cat] = Counter([word for line in data for word in line[1].split(" ") if line[0] == cat])

    for cat in categories:
        for item in copy.copy(cat_list[cat]):
            ratio = cat_list[cat][item]
            ratio = cat_list[cat][item]/category_lengths[cat]

            cat_list[cat][item] = ratio

    return cat_list


def baseline_1(text):
    """
    Implementation of the Baseline 1 task.
    It returns the most common category without regard for the input.

    :param text: a sentence passed on for prediction
    :return: the most common category of the database

    """
    return most_common_category


def baseline_2(text):
    """
    Implementation of the Baseline 2 task.
    It goes through the words of the sentence, determines how often they appeared in each category
    and returns the category with the highest number.

    :param text: a sentence passed on for prediction
    :return: the predicted category based on the number word appearances per category

    """

    max_ratio = 0
    curr_cat = ""
    curr_word = ""
    for word in text.split(" "):
        # Gives around a 0.05 increase in accuracy
        if word not in set(stopwords.words('english')):
            for cat in categories:
                if word in category_ratios[cat]:
                    if category_ratios[cat][word] > max_ratio:
                        max_ratio = category_ratios[cat][word]
                        curr_cat = cat
                        curr_word = word

    if (curr_cat == ""):
        curr_cat = "inform"

    #     print(max_ratio)
    #     print(curr_word)
    return curr_cat

